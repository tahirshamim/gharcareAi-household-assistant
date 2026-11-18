"use strict";
// GharCare AI front end. Safety rule: all dynamic text is inserted as text nodes (never as HTML strings),
// so text from uploaded documents can never run as script. No inline styles: the CSP blocks them.

const state = { appliances: [], tab: "home", chat: [], selected: "", lastQuestion: "" };
const $ = (selector) => document.querySelector(selector);

function h(tag, props, ...children) {
  const node = document.createElement(tag);
  let pendingValue;
  for (const [key, value] of Object.entries(props || {})) {
    if (key === "style") throw new Error("inline style attributes are blocked by the CSP: use a CSS class");
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "value") pendingValue = value;
    else if (key === "checked" || key === "disabled") node[key] = Boolean(value);
    else if (value !== null && value !== undefined && value !== false) node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : String(child));
  }
  if (pendingValue !== undefined && pendingValue !== null) node.value = pendingValue;   // after <option>s exist
  return node;
}

let toastTimer = null;
function toast(message, isError) {
  const el = $("#toast");
  el.textContent = message;
  el.className = isError ? "error" : "";
  el.style.display = "block";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.style.display = "none"; }, isError ? 7000 : 3500);
}

function errorMessage(data, status) {
  if (data && data.error) return data.error;
  if (data && Array.isArray(data.detail)) return data.detail.map((d) => (d.loc || []).slice(1).join(".") + ": " + d.msg).join("; ");
  if (data && typeof data.detail === "string") return data.detail;
  return "Request failed (" + status + ")";
}

function authHeaders(extra) {
  const headers = Object.assign({}, extra || {});
  const token = sessionStorage.getItem("gharcare_token");
  if (token) headers["Authorization"] = "Bearer " + token;
  return headers;
}

async function request(path, options, retried) {
  const response = await fetch("/api" + path, Object.assign({}, options, { headers: authHeaders((options || {}).headers) }));
  if (response.status === 401 && !retried) {
    const token = prompt("This app needs an access token:");
    if (token) { sessionStorage.setItem("gharcare_token", token); return request(path, options, true); }
  }
  return response;
}

async function api(path, options) {
  const response = await request(path, options);
  const text = await response.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (e) { data = { error: text.slice(0, 200) }; }
  if (!response.ok) throw new Error(errorMessage(data, response.status));
  return data;
}

const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

async function downloadFile(path, filename) {
  const response = await request(path);
  if (!response.ok) throw new Error("Download failed (" + response.status + ")");
  saveBlob(await response.blob(), filename);
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = h("a", { href: url, download: filename });
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function guarded(fn) {
  return async (...args) => { try { await fn(...args); } catch (error) { toast(error.message, true); } };
}

// ---------------------------------------------------------------- small helpers
const LABELS = {
  answered: "Answered from your documents", not_found: "Not in your documents", unverified: "Not verified: showing the closest passages",
  unavailable: "AI unavailable: showing the closest passages", needs_clarification: "Which appliance?", safety_escalation: "Safety",
  active: "Active", expiring_soon: "Expiring soon", expired: "Expired", overdue: "Overdue", due_soon: "Due soon", upcoming: "Upcoming",
  completed: "Done", indexed: "Searchable", pending: "Processing", failed: "Could not read", high: "High confidence",
  medium: "Please check", low: "Unsure", manual: "Manual", receipt: "Receipt", warranty_card: "Warranty card", service_invoice: "Service invoice", other: "Other",
};
const label = (id) => LABELS[id] || String(id).replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
const money = (value) => (value === null || value === undefined ? "-" : "PKR " + Number(value).toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 2 }));
const badge = (text) => h("span", { class: "badge s-" + String(text).replace(/\s+/g, "_") }, label(text));
const dash = (value) => (value === null || value === undefined || value === "" ? "-" : value);
const field = (text, input) => h("label", { class: "fld" }, h("span", {}, text), input);
const NUMERIC = new Set(["Amount", "Price", "Cost", "Days left", "Days"]);

function table(headers, rows, emptyText) {
  if (!rows.length) return h("p", { class: "empty" }, emptyText || "Nothing here yet.");
  const cls = (i) => (NUMERIC.has(headers[i]) ? "num" : null);
  return h("table", {}, h("thead", {}, h("tr", {}, headers.map((x, i) => h("th", { class: cls(i) }, x)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, r.map((c, i) => h("td", { class: cls(i) }, c))))));
}

function applianceSelect(selected, blankLabel) {
  const options = state.appliances.map((a) => h("option", { value: a.id }, a.display_name + (a.location ? " (" + a.location + ")" : "")));
  if (blankLabel !== null) options.unshift(h("option", { value: "" }, blankLabel === undefined ? "Not linked" : blankLabel));
  return h("select", { value: selected || "" }, options);
}

const applianceName = (id) => { const a = state.appliances.find((x) => x.id === id); return a ? a.display_name : "-"; };

async function loadAppliances() {
  state.appliances = await api("/appliances");
  const select = $("#appliance-select");
  select.replaceChildren(h("option", { value: "" }, "All appliances"),
    ...state.appliances.map((a) => h("option", { value: a.id }, a.display_name)));
  select.value = state.appliances.some((a) => a.id === state.selected) ? state.selected : "";
  state.selected = select.value;
}

function selectAppliance(id) {
  state.selected = id;
  $("#appliance-select").value = id;
}

// ---------------------------------------------------------------- home: what needs attention
async function renderHome(view) {
  const [maintenance, warranties, drafts, docs] = await Promise.all([api("/maintenance"), api("/warranties"), api("/drafts?status=pending"), api("/documents")]);
  const items = [];
  for (const t of maintenance.overdue) items.push({ level: "urgent", title: t.title + " is overdue", detail: t.appliance + ", due " + t.next_due_date, action: "Open maintenance", tab: "maintenance" });
  for (const t of maintenance.due_soon) items.push({ level: "", title: t.title + " is due in " + t.days_until + " days", detail: t.appliance + ", " + t.next_due_date, action: "Open maintenance", tab: "maintenance" });
  for (const w of warranties.filter((x) => x.status === "expiring_soon")) items.push({ level: "", title: w.appliance + " warranty ends in " + w.days_remaining + " days", detail: "Expires " + w.expiry_date + ". Check anything you want repaired under warranty.", action: "Open warranties", tab: "warranties" });
  for (const w of warranties.filter((x) => !x.verified)) items.push({ level: "info", title: "Confirm the " + w.appliance + " warranty dates", detail: "These were read from a document and have not been checked yet.", action: "Open warranties", tab: "warranties" });
  if (drafts.length) items.push({ level: "info", title: drafts.length + (drafts.length === 1 ? " document is" : " documents are") + " waiting for your review", detail: "Nothing is saved until you check the details.", action: "Review now", tab: "review" });
  for (const d of docs.filter((x) => x.index_status === "failed")) items.push({ level: "urgent", title: d.filename + " could not be read", detail: "Scanned photos are not searchable yet. Upload a text PDF if you have one.", action: "Open documents", tab: "documents" });

  const manuals = docs.filter((d) => d.doc_type === "manual").length;
  view.append(h("p", { class: "summary" }, state.appliances.length + (state.appliances.length === 1 ? " appliance, " : " appliances, ") +
    manuals + (manuals === 1 ? " manual, " : " manuals, ") + docs.length + (docs.length === 1 ? " document in total." : " documents in total.")));

  if (!state.appliances.length) {
    view.append(h("div", { class: "card" }, h("h2", {}, "Get started"), h("ol", { class: "steps" },
      h("li", {}, "Add your appliances: fridge, AC, washing machine."),
      h("li", {}, "Upload their manuals and warranty cards (text PDFs)."),
      h("li", {}, "Ask anything. Answers quote your own documents with the page number.")),
      h("div", { class: "row" }, h("button", { class: "primary", onclick: () => show("appliances") }, "Add an appliance"))));
  } else {
    view.append(h("div", { class: "card" }, h("h2", {}, items.length ? "Needs attention" : "All clear"),
      items.length ? h("ul", { class: "attention" }, items.map((it) => h("li", { class: it.level },
        h("div", { class: "what" }, h("strong", {}, it.title), h("span", {}, it.detail)),
        h("button", { onclick: () => show(it.tab) }, it.action))))
        : h("p", { class: "empty" }, "No overdue maintenance, expiring warranties or documents waiting for review.")));
  }

  const input = h("input", { type: "text", class: "input-grow", placeholder: "e.g. Why is water leaking from the fridge?", "aria-label": "Your question" });
  const go = () => { const q = input.value.trim(); if (q) { show("chat").then(() => sendQuestion(q)); } };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  view.append(h("div", { class: "card" }, h("h2", {}, "Ask about your appliances"),
    h("div", { class: "row" }, input, h("button", { class: "primary", onclick: go }, "Ask"))));
}

// ---------------------------------------------------------------- the conversation
const SUGGESTIONS = ["How do I defrost the freezer?", "Which warranties expire soon?", "Which maintenance is overdue?",
  "How much did I spend on repairs in the last 6 months?", "fridge mein barf jam gayi hai, kya karun?"];

function withCitations(text) {          // "[1]" and "[2, 3]" become small markers; everything stays a text node
  const out = [];
  let last = 0;
  for (const m of text.matchAll(/\[(\d+(?:\s*,\s*\d+)*)\]/g)) {
    out.push(text.slice(last, m.index), h("span", { class: "cite", title: "Source " + m[1] }, m[1].replace(/\s+/g, "")));
    last = m.index + m[0].length;
  }
  out.push(text.slice(last));
  return out;
}

function parseCitation(text) {
  const m = /^(.*?), p\.(\d+)(?: \((.*)\))?$/.exec(text);
  if (!m) return { doc: text, page: null, section: null };
  const section = m[3] && m[3] === m[3].toUpperCase() ? m[3].charAt(0) + m[3].slice(1).toLowerCase() : m[3];
  return { doc: m[1], page: m[2], section };
}

function sourcesBlock(sources) {
  return h("div", { class: "sources" }, h("h4", {}, "Sources"), h("ol", {}, sources.map((s) => {
    const c = parseCitation(s.citation);
    return h("li", {}, h("span", { class: "n" }, String(s.n)), c.page ? h("span", { class: "page" }, "p." + c.page) : null,
      h("span", {}, c.section || ""), h("span", { class: "doc" }, c.doc));
  })));
}

function bubble(message) {
  if (message.role === "user") return h("div", { class: "msg user" }, message.text);
  const parts = message.parts || { body: message.text, records: [], sources: (message.citations || []).map((c, i) => ({ n: i + 1, citation: c })) };
  const safety = message.status === "safety_escalation";
  const node = h("div", { class: "msg bot" + (safety ? " safety" : "") });
  if (safety) node.append(h("h3", {}, "Don't repair this yourself"));
  else if (message.status !== "answered") node.append(h("span", { class: "status badge s-" + message.status }, label(message.status)));
  node.append(h("div", {}, withCitations(parts.body)));
  if (message.status === "needs_clarification" && state.appliances.length) {
    node.append(h("div", { class: "choices" }, state.appliances.map((a) => h("button", { onclick: () => { selectAppliance(a.id); sendQuestion(message.question); } }, a.display_name))));
  }
  if (parts.records.length) node.append(h("div", { class: "records" }, h("h4", {}, "From your records"), h("ul", {}, parts.records.map((r) => h("li", {}, r)))));
  if (parts.sources.length) node.append(sourcesBlock(parts.sources));
  return node;
}

let chatLog = null;
async function sendQuestion(question) {
  if (!chatLog) return;
  state.chat.push({ role: "user", text: question });
  chatLog.append(bubble(state.chat[state.chat.length - 1]));
  const waiting = h("div", { class: "msg pending" }, "Checking your documents and records");
  chatLog.append(waiting);
  waiting.scrollIntoView({ block: "end" });
  try {
    const reply = await post("/ask", { question, appliance_id: state.selected || null });
    state.chat.push({ role: "bot", text: reply.answer, status: reply.status, citations: reply.citations, parts: reply.parts, question });
    waiting.replaceWith(bubble(state.chat[state.chat.length - 1]));
  } catch (error) {
    waiting.remove();
    toast(error.message, true);
  }
  const composer = $(".composer");
  if (composer) composer.scrollIntoView({ block: "end", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
}

function renderChat(view) {
  chatLog = h("div", { class: "chat", "aria-live": "polite" }, state.chat.map(bubble));
  const input = h("input", { type: "text", class: "input-grow", placeholder: "Ask in English or Roman Urdu", "aria-label": "Your question" });
  const send = () => { const q = input.value.trim(); if (!q) return; input.value = ""; suggestions.remove(); sendQuestion(q); };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });
  const suggestions = state.chat.length ? h("span", {}) : h("div", { class: "suggestions" },
    SUGGESTIONS.map((q) => h("button", { onclick: () => { suggestions.remove(); sendQuestion(q); } }, q)));
  view.append(suggestions, chatLog, h("div", { class: "composer" }, input, h("button", { class: "primary", onclick: send }, "Ask")));
  input.focus();
}

// ---------------------------------------------------------------- appliances
async function renderAppliances(view) {
  view.append(h("div", { class: "card" }, h("h2", {}, "Your appliances"), table(
    ["Appliance", "Model", "Serial", "Purchased", "Price", "Location", ""],
    state.appliances.map((a) => [a.display_name, dash(a.model), dash(a.serial_number), dash(a.purchase_date), a.purchase_price ? money(a.purchase_price) : "-", dash(a.location),
      h("button", { class: "danger", onclick: guarded(async () => {
        if (!confirm("Delete " + a.display_name + " and its warranties, repairs and tasks? Uploaded documents are kept.")) return;
        await api("/appliances/" + a.id, { method: "DELETE" });
        await loadAppliances();
        show("appliances");
      }) }, "Delete")]), "No appliances yet. Add your first one below.")));
  const f = {};
  const input = (name, text, placeholder, type) => { f[name] = h("input", { type: type || "text", placeholder: placeholder || "" }); return field(text, f[name]); };
  const add = guarded(async () => {
    const body = {};
    for (const [name, el] of Object.entries(f)) body[name] = el.value.trim() || null;
    await post("/appliances", body);
    toast("Appliance added");
    await loadAppliances();
    show("appliances");
  });
  view.append(h("div", { class: "card" }, h("h2", {}, "Add an appliance"), h("div", { class: "form" },
    input("name", "Name", "Fridge"), input("brand", "Brand", "Haier"), input("model", "Model", "HRF-306"), input("serial_number", "Serial number"),
    input("purchase_date", "Purchase date", "", "date"), input("purchase_price", "Price (PKR)", "95000"), input("location", "Location", "Kitchen"),
    h("button", { class: "primary", onclick: add }, "Add appliance"))));
}

// ---------------------------------------------------------------- documents
const EXTRACTABLE = ["receipt", "warranty_card", "service_invoice"];

async function renderDocuments(view) {
  const docs = await api("/documents");
  const file = h("input", { type: "file", accept: ".pdf,.txt,.md" });
  const type = h("select", { value: "manual" }, ["manual", "receipt", "warranty_card", "service_invoice", "other"].map((t) => h("option", { value: t }, label(t))));
  const owner = applianceSelect(state.selected);
  const upload = guarded(async () => {
    if (!file.files.length) { toast("Choose a file first", true); return; }
    const form = new FormData();
    form.append("file", file.files[0]);
    form.append("doc_type", type.value);
    form.append("appliance_id", owner.value);
    const result = await api("/documents", { method: "POST", body: form });
    toast(result.warning || (result.duplicate ? "This file was already uploaded." : "Uploaded. " + result.chunk_count + " passages are now searchable."), Boolean(result.warning));
    show("documents");
  });
  view.append(h("div", { class: "card" }, h("h2", {}, "Upload a document"),
    h("p", { class: "muted" }, "Text-based PDFs, .txt and .md files. Scanned photos are stored but not searchable yet."),
    h("div", { class: "form" }, field("File", file), field("Type", type), field("Appliance", owner), h("button", { class: "primary", onclick: upload }, "Upload"))));
  view.append(h("div", { class: "card" }, h("h2", {}, "Your documents"), table(["File", "Type", "Appliance", "Status", ""],
    docs.map((d) => [d.filename, label(d.doc_type), applianceName(d.appliance_id), badge(d.index_status),
      h("span", { class: "row" },
        EXTRACTABLE.includes(d.doc_type) ? h("button", { onclick: guarded(async () => { await post("/documents/" + d.id + "/extract"); toast("Details read. Please review them."); show("review"); }) }, "Read details") : null,
        h("button", { onclick: guarded(() => downloadFile("/documents/" + d.id + "/file", d.filename)) }, "Download"),
        h("button", { class: "danger", onclick: guarded(async () => { if (!confirm("Delete " + d.filename + "?")) return; await api("/documents/" + d.id, { method: "DELETE" }); show("documents"); }) }, "Delete"))]),
    "No documents yet. Upload a manual to start asking questions about it.")));
}

// ---------------------------------------------------------------- review of extracted details
function draftCard(draft) {
  const rows = draft.fields.map((f) => {
    const input = f.kind === "date" ? h("input", { type: "date", value: f.value || "" })
      : f.kind === "months" ? h("input", { type: "number", min: 1, max: 120, value: f.value || "" })
      : h("input", { type: "text", value: f.value || "", class: "input-wide" });
    const needsCheck = f.value !== null && f.confidence !== "high";
    const check = needsCheck ? h("input", { type: "checkbox" }) : null;
    input.addEventListener("input", () => { if (check) check.checked = true; });
    const alternatives = f.alternatives.map((a) => h("button", { type: "button", onclick: () => { input.value = a; if (check) check.checked = true; } }, "Use " + a));
    const why = [f.note, f.evidence ? 'Document says: "' + f.evidence + '"' : null].filter(Boolean).join(" | ");
    return { f, input, check, node: h("div", { class: "field" },
      h("div", {}, h("strong", {}, f.label), " ", badge(f.confidence)),
      h("div", { class: "row" }, input, check ? h("label", {}, check, " I checked this") : null, alternatives),
      why ? h("div", { class: "why" }, why) : null) };
  });
  const owner = applianceSelect(state.selected, "Detect or create automatically");
  const confirmBtn = h("button", { class: "primary", onclick: guarded(async () => {
    const values = {}, reviewed = [];
    for (const r of rows) { values[r.f.name] = r.input.value.trim() || null; if (r.check && r.check.checked) reviewed.push(r.f.name); }
    const result = await post("/drafts/" + draft.id + "/confirm", { values, reviewed_fields: reviewed, appliance_id: owner.value || null });
    toast("Saved. " + (result.changes.length ? result.changes.join("; ") : "Records created."));
    await loadAppliances();
    show("review");
  }) }, "Save details");
  const rejectBtn = h("button", { onclick: guarded(async () => { await post("/drafts/" + draft.id + "/reject"); show("review"); }) }, "Discard");
  return h("div", { class: "card" }, h("h2", {}, label(draft.doc_type)),
    h("p", { class: "muted" }, "Check each value against your document. Values marked \"Please check\" or \"Unsure\" need your tick before anything is saved."),
    rows.map((r) => r.node), h("div", { class: "row" }, field("Belongs to", owner), confirmBtn, rejectBtn));
}

async function renderReview(view) {
  const drafts = await api("/drafts?status=pending");
  view.append(drafts.length ? drafts.map(draftCard) : h("div", { class: "card quiet" }, h("p", { class: "empty" },
    "Nothing to review. Upload a receipt, warranty card or service invoice, then press \"Read details\" on the Documents page.")));
  const accuracy = await api("/extraction/accuracy");
  if (accuracy.drafts) view.append(h("p", { class: "muted" }, "Reading accuracy so far: " + accuracy.fields_correct + " of " + accuracy.fields_read +
    " values were kept as read (" + Math.round(100 * accuracy.accuracy) + "%), across " + accuracy.drafts + " confirmed document(s)."));
}

// ---------------------------------------------------------------- warranties, maintenance, repairs
async function renderWarranties(view) {
  const rows = await api("/warranties");
  view.append(h("div", { class: "card" }, h("h2", {}, "Warranties"), table(["Appliance", "Starts", "Expires", "Days left", "Status", "Provider", ""],
    rows.map((w) => [w.appliance, w.start_date, w.expiry_date, String(w.days_remaining), badge(w.status), dash(w.provider),
      w.verified ? h("span", { class: "muted" }, "Confirmed") : h("button", { onclick: guarded(async () => { await post("/warranties/" + w.id + "/confirm"); show("warranties"); }) }, "Confirm dates")]),
    "No warranties yet. Upload a warranty card and read its details, or add one below.")));
  const owner = applianceSelect(state.selected, null), start = h("input", { type: "date" }), months = h("input", { type: "number", min: 1, max: 120, placeholder: "12", class: "input-cost" }), provider = h("input", { type: "text", placeholder: "Dealer or brand" });
  const add = guarded(async () => { await post("/warranties", { appliance_id: owner.value, start_date: start.value, duration_months: Number(months.value), provider: provider.value || null }); toast("Warranty added"); show("warranties"); });
  view.append(h("div", { class: "card" }, h("h2", {}, "Add a warranty"), h("div", { class: "form" },
    field("Appliance", owner), field("Starts", start), field("Months", months), field("Provider", provider), h("button", { class: "primary", onclick: add }, "Add warranty"))));
}

async function renderMaintenance(view) {
  const data = await api("/maintenance" + (state.selected ? "?appliance_id=" + encodeURIComponent(state.selected) : ""));
  const group = (title, items) => items.length ? h("div", { class: "card" }, h("h2", {}, title), table(["Appliance", "Task", "Due", "Days", "Repeats", ""],
    items.map((t) => { const cost = h("input", { type: "text", placeholder: "Cost (PKR)", class: "input-cost", "aria-label": "Cost in PKR" });
      return [t.appliance, t.title, dash(t.next_due_date), dash(t.days_until === null || t.days_until === undefined ? null : String(t.days_until)), t.interval_months ? "Every " + t.interval_months + " months" : "Once",
        t.status === "completed" ? "" : h("span", { class: "row" }, cost, h("button", { onclick: guarded(async () => { await post("/maintenance/" + t.id + "/complete", { cost: cost.value.trim() || null }); toast("Marked done"); show("maintenance"); }) }, "Mark done"))]; }))) : null;
  view.append(group("Overdue", data.overdue), group("Due soon", data.due_soon), group("Upcoming", data.upcoming), group("Done", data.completed));
  if (!data.overdue.length && !data.due_soon.length && !data.upcoming.length && !data.completed.length) view.append(h("div", { class: "card quiet" }, h("p", { class: "empty" }, "No maintenance tasks yet. Add one below, for example an AC service every 6 months.")));
  const owner = applianceSelect(state.selected, null), title = h("input", { type: "text", placeholder: "AC service" }), due = h("input", { type: "date" }), every = h("input", { type: "number", min: 1, max: 120, placeholder: "6", class: "input-cost" });
  view.append(h("div", { class: "card" }, h("h2", {}, "Add a task"), h("div", { class: "form" }, field("Appliance", owner), field("Task", title), field("Due", due), field("Repeat every (months)", every),
    h("button", { class: "primary", onclick: guarded(async () => { await post("/maintenance", { appliance_id: owner.value, title: title.value, next_due_date: due.value, interval_months: every.value ? Number(every.value) : null }); toast("Task added"); show("maintenance"); }) }, "Add task"))));
}

async function renderRepairs(view) {
  const rows = await api("/repairs" + (state.selected ? "?appliance_id=" + encodeURIComponent(state.selected) : ""));
  view.append(h("div", { class: "card" }, h("h2", {}, "Repair history"), table(["Date", "Appliance", "Problem", "What was done", "Cost"],
    rows.map((r) => [r.reported_date, applianceName(r.appliance_id), r.problem_description, dash(r.resolution), Number(r.cost) > 0 ? money(r.cost) : "Not recorded"]),
    "No repairs recorded. Past repairs help the assistant spot repeat problems.")));
  const owner = applianceSelect(state.selected, null), date = h("input", { type: "date" }), problem = h("input", { type: "text", placeholder: "Ice build-up in freezer", class: "input-mid" }),
    resolution = h("input", { type: "text", placeholder: "Replaced defrost sensor" }), cost = h("input", { type: "text", placeholder: "3500", class: "input-cost" });
  view.append(h("div", { class: "card" }, h("h2", {}, "Add a repair"), h("div", { class: "form" }, field("Appliance", owner), field("Date", date), field("Problem", problem), field("What was done", resolution), field("Cost (PKR)", cost),
    h("button", { class: "primary", onclick: guarded(async () => { await post("/repairs", { appliance_id: owner.value, reported_date: date.value, problem_description: problem.value, resolution: resolution.value || null, cost: cost.value || null }); toast("Repair added"); show("repairs"); }) }, "Add repair"))));
}

// ---------------------------------------------------------------- expenses and report
async function renderExpenses(view) {
  const months = h("select", { value: "6" }, ["3", "6", "12", "24"].map((m) => h("option", { value: m }, "Last " + m + " months")));
  const category = h("select", { value: "" }, [["", "All spending"], ["repair", "Repairs"], ["maintenance", "Maintenance"], ["purchase", "Purchases"]].map(([v, l]) => h("option", { value: v }, l)));
  const out = h("div", {});
  const load = guarded(async () => {
    const q = "?months=" + months.value + (category.value ? "&category=" + category.value : "") + (state.selected ? "&appliance_id=" + encodeURIComponent(state.selected) : "");
    const e = await api("/expenses" + q);
    const c = e.comparison;
    const change = c.pct_change === null ? "" : (c.pct_change >= 0 ? ", up " : ", down ") + Math.abs(c.pct_change) + "%";
    const gaps = e.gaps.records_without_cost + e.gaps.purchases_without_price + e.gaps.purchases_without_date;
    out.replaceChildren(...[                       // replaceChildren would print a null as the text "null"
      h("p", { class: "total" }, money(e.total)),
      h("p", { class: "muted" }, e.since + " to " + e.until + ". The period before: " + money(c.previous_total) + change + "."),
      gaps ? h("p", { class: "muted" }, "Totals may be understated: " + e.gaps.records_without_cost + " records have no cost, " + e.gaps.purchases_without_price + " purchases no price, " + e.gaps.purchases_without_date + " no date.") : null,
      h("div", { class: "split" },
        h("div", {}, h("h3", {}, "By type"), table(["Type", "Amount"], Object.entries(e.by_category).map(([k, v]) => [label(k), money(v)]), "No spending in this period.")),
        h("div", {}, h("h3", {}, "By appliance"), table(["Appliance", "Amount"], e.by_appliance.map((a) => [a.appliance, money(a.total)]), "No spending in this period."))),
      h("h3", {}, "By month"), table(["Month", "Amount"], Object.entries(e.by_month).map(([k, v]) => [k, money(v)]), "No spending in this period."),
      h("h3", {}, "Every payment"), table(["Date", "Type", "Appliance", "Amount", "What"], e.lines.map((l) => [l.date, label(l.category), l.appliance, money(l.amount), l.description]), "No payments in this period.")].filter(Boolean));
  });
  months.addEventListener("change", load); category.addEventListener("change", load);
  view.append(h("div", { class: "card" }, h("div", { class: "form" }, field("Period", months), field("Show", category)), out));
  await load();
}

function renderReport(view) {
  const months = h("select", { value: "6" }, ["3", "6", "12", "24"].map((m) => h("option", { value: m }, "Last " + m + " months")));
  const out = h("div", {});
  view.append(h("div", { class: "card" }, h("div", { class: "form" }, field("Period", months), h("button", { class: "primary", onclick: guarded(async () => {
      const r = await post("/report?months=" + months.value);
      out.replaceChildren(h("div", { class: "row" }, h("button", { onclick: () => saveBlob(new Blob([r.markdown], { type: "text/markdown" }), "household-report.md") }, "Download report")), h("pre", {}, r.markdown));
    }) }, "Create report")), out));
}

// ---------------------------------------------------------------- shell
const TABS = [
  ["home", "Home", renderHome, "", "Your household at a glance: what needs attention, and a quick way to ask."],
  ["chat", "Ask", renderChat, "", "Answers come only from your own manuals and records, with the page they came from. If something isn't there, you'll be told."],
  ["appliances", "Appliances", renderAppliances, "Your home", "Everything in the house you want to keep track of."],
  ["documents", "Documents", renderDocuments, "Your home", "Manuals, receipts, warranty cards and invoices. Originals are kept unchanged."],
  ["review", "Review", renderReview, "Your home", "Details read from receipts and warranty cards. Nothing is saved until you check it."],
  ["warranties", "Warranties", renderWarranties, "Care", "When each warranty ends. Coverage for a specific fault depends on the card's terms."],
  ["maintenance", "Maintenance", renderMaintenance, "Care", "Regular servicing, so small problems don't become repairs."],
  ["repairs", "Repairs", renderRepairs, "Care", "What went wrong, what was done, and what it cost."],
  ["expenses", "Expenses", renderExpenses, "Money", "Totals are exact, from the costs you recorded."],
  ["report", "Report", renderReport, "Money", "A household summary built only from your saved records. It lists any gaps."],
];

function renderNav(tab) {
  const groups = [];
  for (const [id, text, , group] of TABS) {
    let g = groups.find((x) => x.name === group);
    if (!g) { g = { name: group, items: [] }; groups.push(g); }
    g.items.push(h("button", { class: id === tab ? "active" : "", "aria-current": id === tab ? "page" : null, onclick: () => show(id) }, text));
  }
  $("#tabs").replaceChildren(...groups.map((g) => h("div", { class: "nav-group" }, g.name ? h("p", { class: "nav-label" }, g.name) : null, g.items)));
}

async function show(tab) {
  state.tab = tab;
  if (tab !== "chat") chatLog = null;
  renderNav(tab);
  const [, title, render, , lede] = TABS.find((t) => t[0] === tab);
  const view = $("#view");
  view.replaceChildren(h("h1", { class: "page-title" }, title === "Ask" ? "Ask about your appliances" : title), h("p", { class: "lede" }, lede));
  try { await render(view); } catch (error) { toast(error.message, true); }
}

async function start() {
  $("#appliance-select").addEventListener("change", (e) => { state.selected = e.target.value; if (["maintenance", "repairs", "expenses"].includes(state.tab)) show(state.tab); });
  try {
    const health = await api("/health");
    const mode = $("#mode");
    mode.textContent = health.llm_configured ? "AI ready" : "AI off: records and manual passages only";
    mode.classList.add(health.llm_configured ? "on" : "off");
    await loadAppliances();
  } catch (error) { toast(error.message, true); }
  show("home");
}

start();
