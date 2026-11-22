"use strict";
// Audit environment UI. Plain DOM, no dependencies, works offline.
// Every value from the server or an imported folder is written with textContent, never as HTML.

const TYPES = {
  duplicate_invoice: "Дубль счёта",
  payment_allocation: "Распределение платежа",
  fx_rate: "Курс или дата курса",
  period_cutoff: "Отчётный период",
  vat_on_top: "НДС сверху цены",
  uncollected_service: "Продление без оплаты",
};
const INTERACTIONS = {
  fx_boundary_payment: "Валютная оплата на границе периода",
  fx_partial_allocation: "Валютная оплата, часть в авансах",
};
const PRESETS = {
  default: { errors: { duplicate_invoice: 1, payment_allocation: 1, fx_rate: 1, period_cutoff: 1 }, interactions: {} },
  all: { errors: { duplicate_invoice: 1, payment_allocation: 1, fx_rate: 1, period_cutoff: 1, vat_on_top: 1, uncollected_service: 1 },
         interactions: { fx_boundary_payment: 1, fx_partial_allocation: 1 } },
  interacting: { errors: { duplicate_invoice: 0, payment_allocation: 0, fx_rate: 0, period_cutoff: 0 },
                 interactions: { fx_boundary_payment: 1, fx_partial_allocation: 1 } },
  practice: { errors: { duplicate_invoice: 0, payment_allocation: 0, fx_rate: 0, period_cutoff: 0, vat_on_top: 2, uncollected_service: 1 },
              interactions: {} },
  clean: { errors: { duplicate_invoice: 0, payment_allocation: 0, fx_rate: 0, period_cutoff: 0 }, interactions: {} },
};
const TABLE_IDS = { ledger: "row_id", invoices: "invoice_id", payments: "payment_id", fx: "fx_id" };
const LEDGER_PAGE = 100;

const state = {
  session: null, graded: false, training: false, files: null, observation: null,
  docs: [], byId: new Map(), selectedDoc: null, ledgerPage: 0, marked: new Set(),
  drafts: [], activeDraft: 0, finished: false,
};

// ---- helpers --------------------------------------------------------------
const $ = (id) => document.getElementById(id);

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); return node; }

function minorUnits(currency) {
  const units = state.files && state.files.manifest.minor_units;
  return units && Number.isInteger(units[currency]) ? units[currency] : 2;
}

function money(minor, currency) {
  if (typeof minor !== "number") return "—";
  const digits = minorUnits(currency);
  const value = minor / 10 ** digits;
  return value.toLocaleString("ru-RU", { minimumFractionDigits: digits, maximumFractionDigits: digits }) + (currency ? " " + currency : "");
}

function dateText(iso) {
  if (!iso) return "—";
  return String(iso).replace("T", " ").replace(/:\d\d([+-]\d\d:\d\d)$/, " $1");
}

function status(text, isError = false) {
  const node = $("status");
  node.textContent = text || "";
  node.className = "status" + (isError ? " error" : "");
}

async function api(path, body) {
  const options = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const response = await fetch(path, options);
  let data;
  try { data = await response.json(); } catch { data = { error: `HTTP ${response.status}` }; }
  if (!response.ok) {
    const problems = Array.isArray(data.problems) ? ": " + data.problems.slice(0, 5).join("; ") : "";
    throw new Error((data.error || `HTTP ${response.status}`) + problems);
  }
  return data;
}

async function busy(button, work) {
  button.disabled = true;
  try { await work(); } catch (error) { status(error.message, true); } finally { button.disabled = false; }
}

function table(node, columns, rows, { onRow, rowClass } = {}) {
  clear(node);
  node.append(el("thead", {}, el("tr", {}, columns.map((c) => el("th", { class: c.num ? "num" : null, text: c.title })))));
  const body = el("tbody");
  for (const row of rows) {
    const tr = el("tr", { class: rowClass ? rowClass(row) : null });
    for (const c of columns) {
      const value = c.value(row);
      tr.append(el("td", { class: c.num ? "num" : c.id ? "id" : c.wrap ? "wrap" : null }, value instanceof Node ? value : String(value ?? "—")));
    }
    if (onRow) tr.addEventListener("click", () => onRow(row));
    body.append(tr);
  }
  node.append(body);
}

function card(key, value, tone) {
  return el("div", { class: "card" + (tone ? " " + tone : "") }, el("div", { class: "k", text: key }), el("div", { class: "v", text: value }));
}

// ---- tabs -----------------------------------------------------------------
function showTab(name) {
  for (const button of $("tabs").querySelectorAll("button")) {
    button.setAttribute("aria-selected", String(button.dataset.tab === name));
  }
  for (const section of document.querySelectorAll(".tab")) section.hidden = section.id !== "tab-" + name;
}

// ---- episode creation -------------------------------------------------------
function buildErrorInputs() {
  const box = clear($("error-counts"));
  for (const [kind, title] of Object.entries(TYPES)) {
    box.append(el("label", {}, title, el("input", { type: "number", min: 0, max: 6, value: kind in PRESETS.default.errors ? 1 : 0, "data-error": kind })));
  }
  for (const [kind, title] of Object.entries(INTERACTIONS)) {
    box.append(el("label", {}, title, el("input", { type: "number", min: 0, max: 3, value: 0, "data-interaction": kind })));
  }
}

function errorConfig() {
  const preset = $("preset").value;
  if (preset !== "custom") return PRESETS[preset];
  const errors = {}, interactions = {};
  for (const input of $("error-counts").querySelectorAll("input")) {
    const value = Number.parseInt(input.value, 10) || 0;
    if (input.dataset.error) errors[input.dataset.error] = value;
    else interactions[input.dataset.interaction] = value;
  }
  return { errors, interactions };
}

async function createEpisode(event) {
  event.preventDefault();
  const seed = Number.parseInt($("seed").value, 10);
  if (!Number.isInteger(seed) || seed < 0) { status("Seed должен быть целым числом ≥ 0", true); return; }
  const mode = $("mode").value;
  const config = { keyspace: mode === "closed" ? "secret" : "open", difficulty: $("difficulty").value, ...errorConfig() };
  await busy($("create-btn"), async () => {
    status("Создаю…");
    const data = await api("/api/episodes", { seed, mode, config });
    await openSession(data, mode === "open");
    status(`Готово · seed ${seed} · ${config.difficulty} · ${mode === "open" ? "открытый" : "закрытый"}`);
  });
}

async function importFolder(event) {
  event.preventDefault();
  const folder = $("folder").value.trim();
  if (!folder) { status("Укажите путь к папке", true); return; }
  await busy($("import-btn"), async () => {
    status("Проверяю файлы…");
    const data = await api("/api/episodes/import", { folder });
    await openSession(data, false);
    status("Папка открыта · без эталона, оценки нет");
  });
}

async function openSession(data, training) {
  state.session = data.session_id;
  state.graded = data.graded;
  state.training = training;
  state.observation = data.observation;
  state.files = await api(`/api/sessions/${data.session_id}/files`);
  state.marked = new Set();
  state.selectedDoc = null;
  state.ledgerPage = 0;
  state.drafts = [];
  state.activeDraft = 0;
  state.finished = false;
  indexFiles();
  renderHeader();
  renderOverview();
  renderDocuments();
  renderLedgerFilters();
  renderLedger();
  renderFx();
  renderPolicy();
  resetManual();
  $("agent-result").hidden = true;
  $("agent-empty").hidden = false;
  $("empty").hidden = true;
  $("workspace").hidden = false;
  $("tabs").hidden = false;
  $("agent-panel").hidden = false;
  showTab("overview");
}

function indexFiles() {
  const { tables, documents } = state.files;
  state.byId = new Map();
  state.docs = documents.map((doc) => ({ ...doc, kind: doc.invoice_id ? "invoice" : "payment" }));
  for (const doc of state.docs) {
    state.byId.set(doc.doc_id, { tab: "documents", doc });
    state.byId.set(doc.invoice_id || doc.payment_id, { tab: "documents", doc });
  }
  for (const row of tables.ledger) state.byId.set(row.row_id, { tab: "ledger", row });
  for (const row of tables.fx) state.byId.set(row.fx_id, { tab: "fx", row });
  state.entityNames = new Map();
  for (const doc of state.docs) {
    const party = doc.counterparty || doc.customer || doc.buyer;
    if (party && party.entity_id && party.name) state.entityNames.set(party.entity_id, party.name);
  }
}

function entityName(id) { return state.entityNames.get(id) || id || "—"; }

// ---- header and overview ------------------------------------------------------
function renderHeader() {
  const m = state.files.manifest;
  $("episode-info").hidden = false;
  const badge = $("mode-badge");
  badge.textContent = !state.graded ? "Свои документы" : state.training ? "Открытый режим" : "Закрытый режим";
  badge.className = "badge" + (state.training ? "" : " closed");
  $("company").textContent = (m.company && m.company.name) || m.episode_id;
  $("period").textContent = `${String(m.period_start).slice(0, 10)} — ${String(m.period_end).slice(0, 10)} (${m.timezone})`;
  $("steps-left").textContent = state.observation.remaining_steps ?? 100;
}

function renderOverview() {
  const m = state.files.manifest;
  const cards = clear($("overview-cards"));
  cards.append(
    card("Проводки", m.counts.ledger),
    card("Счета", m.counts.invoices),
    card("Платежи", m.counts.payments),
    card("Курсы", m.counts.fx),
    card("Валюты", Object.keys(m.minor_units).join(", ")),
    card("База", m.base_currency),
  );
  $("task-text").textContent = state.observation.task || "—";
  const rules = (state.files.policy && state.files.policy.findings) || {};
  const rows = Object.keys(TYPES).map((kind) => ({ kind, rule: rules[kind] || {} }));
  table($("types-table"), [
    { title: "Что ищем", value: (r) => TYPES[r.kind] },
    { title: "Код", value: (r) => el("code", { class: "muted", text: r.kind }) },
    { title: "Указать", value: (r) => r.rule.affected_ids || "—", wrap: true },
    { title: "Допуск", value: (r) => (r.rule.tolerance_minor ?? "—"), num: true },
  ], rows);
}

// ---- documents ------------------------------------------------------------------
function docTitle(doc) {
  return doc.title || doc.number || (doc.kind === "invoice" ? "Счёт" : "Платёж");
}

function docShort(doc) {
  const kind = doc.kind === "payment" ? "Платёж" : doc.document_kind === "waybill" ? "Накладная" : "Акт";
  return doc.number ? `${kind} ${doc.number}` : kind;
}

function docAmount(doc) {
  return doc.kind === "invoice" ? money(doc.total_minor, doc.currency) : money(doc.amount_minor, doc.currency);
}

function renderDocuments() {
  const query = $("doc-search").value.trim().toLowerCase();
  const kind = $("doc-kind").value;
  const rows = state.docs.filter((doc) => (kind === "all" || doc.kind === kind) &&
    (!query || JSON.stringify(doc).toLowerCase().includes(query) || entityName(doc.entity_id).toLowerCase().includes(query)));
  rows.sort((a, b) => String(a.issued_at || a.timestamp).localeCompare(String(b.issued_at || b.timestamp)));
  table($("doc-table"), [
    { title: "Дата", value: (d) => dateText(d.issued_at || d.timestamp).slice(0, 16) },
    { title: "Документ", value: (d) => docShort(d) },
    { title: "Контрагент", value: (d) => entityName(d.entity_id || (d.counterparty && d.counterparty.entity_id)) },
    { title: "Сумма", value: docAmount, num: true },
  ], rows, {
    onRow: (doc) => selectDoc(doc),
    rowClass: (doc) => [doc === state.selectedDoc ? "selected" : "", state.marked.has(doc.doc_id) ? "mark" : ""].join(" ").trim() || null,
  });
}

function idButton(id) {
  return el("button", { class: "link", type: "button", text: id, onclick: (e) => { e.stopPropagation(); jumpTo(id); } });
}

function selectDoc(doc) {
  state.selectedDoc = doc;
  renderDocuments();
  const box = clear($("doc-detail"));
  const meta = el("dl", { class: "meta" });
  const add = (k, v) => { if (v !== undefined && v !== null && v !== "") meta.append(el("dt", { text: k }), el("dd", {}, v)); };
  add("doc_id", el("code", { text: doc.doc_id }));
  if (doc.kind === "invoice") {
    add("invoice_id", el("code", { text: doc.invoice_id }));
    add("Клиент", entityName(doc.entity_id));
    add("Выставлен", dateText(doc.issued_at));
    add("Срок оплаты", dateText(doc.due_at));
    add("Сумма", docAmount(doc));
    add("Договор", doc.contract);
    add("Менеджер", doc.manager);
  } else {
    add("payment_id", el("code", { text: doc.payment_id }));
    add("Плательщик", entityName(doc.counterparty && doc.counterparty.entity_id));
    add("Время операции", dateText(doc.timestamp));
    add("Сумма", docAmount(doc));
    add("Назначение", doc.purpose);
    add("Ссылки на счета", doc.invoice_refs ? el("span", {}, doc.invoice_refs.map((ref) => [idButton(ref), " "])) : "—");
  }
  const node = el("div", { class: "doc" + (state.marked.has(doc.doc_id) ? " mark" : "") },
    el("h4", { text: docTitle(doc) }), meta);
  if (Array.isArray(doc.line_items)) {
    const lines = el("table", { class: "data" });
    table(lines, [
      { title: "№", value: (l) => l.line_no },
      { title: "Услуга или товар", value: (l) => l.description, wrap: true },
      { title: "Период", value: (l) => `${l.service_period_start} — ${l.service_period_end}` },
      { title: "Кол-во", value: (l) => l.quantity, num: true },
      { title: "НДС", value: (l) => (l.vat_rate ? `${Number(l.vat_rate) * 100}% · ${money(l.vat_minor, doc.currency)}` : "без НДС"), num: true },
      { title: "Итого", value: (l) => money(l.total_minor, doc.currency), num: true },
    ], doc.line_items);
    node.append(el("div", { class: "table-wrap" }, lines));
  }
  const rows = state.files.tables.ledger.filter((r) => r.invoice_id === doc.invoice_id && doc.invoice_id || r.payment_id === doc.payment_id && doc.payment_id);
  if (rows.length) {
    node.append(el("h3", { text: `Проводки по документу (${rows.length})` }));
    const t = el("table", { class: "data" });
    ledgerTable(t, rows);
    node.append(el("div", { class: "table-wrap" }, t));
  }
  if (doc.vat_invoice) node.append(el("p", { class: "hint", text: `Приложен ЭСФ, НДС ${money(doc.vat_invoice.vat_minor, doc.currency)}.` }));
  box.append(node);
}

// ---- ledger ---------------------------------------------------------------------
function renderLedgerFilters() {
  const select = $("ledger-account");
  const accounts = [...new Set(state.files.tables.ledger.map((r) => r.account))].sort();
  const names = (state.files.policy && state.files.policy.accounts) || {};
  clear(select).append(el("option", { value: "", text: "Все счета учёта" }));
  for (const account of accounts) select.append(el("option", { value: account, text: `${account} ${names[account] || ""}`.trim() }));
}

function ledgerTable(node, rows) {
  table(node, [
    { title: "row_id", value: (r) => r.row_id, id: true },
    { title: "Время", value: (r) => dateText(r.timestamp) },
    { title: "Счёт", value: (r) => r.account },
    { title: "Дебет", value: (r) => (r.debit_minor ? money(r.debit_minor) : ""), num: true },
    { title: "Кредит", value: (r) => (r.credit_minor ? money(r.credit_minor) : ""), num: true },
    { title: "Контрагент", value: (r) => (r.entity_id ? entityName(r.entity_id) : "—") },
    { title: "Счёт-документ", value: (r) => (r.invoice_id ? idButton(r.invoice_id) : "—") },
    { title: "Платёж", value: (r) => (r.payment_id ? idButton(r.payment_id) : "—") },
  ], rows, { rowClass: (r) => (state.marked.has(r.row_id) ? "mark" : null) });
}

function filteredLedger() {
  const query = $("ledger-search").value.trim().toLowerCase();
  const account = $("ledger-account").value;
  return state.files.tables.ledger.filter((r) => (!account || r.account === account) &&
    (!query || Object.values(r).some((v) => String(v).toLowerCase().includes(query)) || entityName(r.entity_id).toLowerCase().includes(query)));
}

function renderLedger() {
  const rows = filteredLedger();
  const pages = Math.max(1, Math.ceil(rows.length / LEDGER_PAGE));
  state.ledgerPage = Math.min(state.ledgerPage, pages - 1);
  const start = state.ledgerPage * LEDGER_PAGE;
  ledgerTable($("ledger-table"), rows.slice(start, start + LEDGER_PAGE));
  $("ledger-count").textContent = `${rows.length} проводок`;
  const pager = clear($("ledger-pager"));
  if (pages > 1) {
    pager.append(
      el("button", { class: "small", text: "←", disabled: state.ledgerPage === 0, onclick: () => { state.ledgerPage--; renderLedger(); } }),
      el("span", { class: "muted", text: ` стр. ${state.ledgerPage + 1} из ${pages} ` }),
      el("button", { class: "small", text: "→", disabled: state.ledgerPage >= pages - 1, onclick: () => { state.ledgerPage++; renderLedger(); } }),
    );
  }
}

// ---- fx and policy ----------------------------------------------------------------
function renderFx() {
  const query = $("fx-search").value.trim().toLowerCase();
  const rows = state.files.tables.fx.filter((r) => !query || Object.values(r).some((v) => String(v).toLowerCase().includes(query)))
    .sort((a, b) => (a.date + a.source_currency).localeCompare(b.date + b.source_currency));
  table($("fx-table"), [
    { title: "Дата", value: (r) => r.date },
    { title: "Валюта", value: (r) => `${r.source_currency} → ${r.target_currency}` },
    { title: "Курс", value: (r) => r.rate, num: true },
    { title: "fx_id", value: (r) => r.fx_id, id: true },
  ], rows, { rowClass: (r) => (state.marked.has(r.fx_id) ? "mark" : null) });
}

function renderValue(value) {
  if (value === null || typeof value !== "object") return el("span", { text: String(value) });
  if (Array.isArray(value)) return el("span", { text: value.map((v) => (typeof v === "object" ? JSON.stringify(v) : String(v))).join(", ") });
  const dl = el("dl", { class: "kv" });
  for (const [k, v] of Object.entries(value)) dl.append(el("dt", { text: k }), el("dd", {}, renderValue(v)));
  return dl;
}

function renderPolicy() {
  const box = clear($("policy-view"));
  for (const [section, value] of Object.entries(state.files.policy || {})) {
    if (section === "schema_version") continue;
    box.append(el("details", { class: "policy", open: section === "period" }, el("summary", { text: section }), renderValue(value)));
  }
}

// ---- evidence jump ------------------------------------------------------------------
function jumpTo(id) {
  const target = state.byId.get(id);
  if (!target) { status(`ID ${id} не найден в открытых таблицах`, true); return; }
  if (target.tab === "documents") {
    $("doc-search").value = "";
    $("doc-kind").value = "all";
    showTab("documents");
    selectDoc(target.doc);
  } else if (target.tab === "ledger") {
    $("ledger-search").value = id;
    $("ledger-account").value = "";
    state.ledgerPage = 0;
    renderLedger();
    showTab("ledger");
  } else {
    $("fx-search").value = id;
    renderFx();
    showTab("fx");
  }
}

function markIds(ids) {
  state.marked = new Set(ids);
  renderDocuments();
  renderLedger();
  renderFx();
}

function findingCard(finding, verdict) {
  const ids = el("div", { class: "ids" });
  for (const id of finding.affected_ids || []) ids.append(el("button", { class: "chip", type: "button", title: "Затронутый ID", text: "◎ " + id, onclick: () => jumpTo(id) }));
  for (const id of finding.evidence_ids || []) ids.append(el("button", { class: "chip", type: "button", title: "Доказательство", text: id, onclick: () => jumpTo(id) }));
  return el("div", { class: "finding" },
    el("div", { class: "head" },
      el("span", { text: TYPES[finding.type] || finding.type }),
      el("span", {}, money(finding.claimed_amount_minor, finding.currency), verdict || "")),
    ids,
    finding.explanation ? el("p", { class: "muted", text: finding.explanation }) : null,
    el("button", { class: "small", type: "button", text: "Подсветить в таблицах", onclick: () => markIds([...(finding.affected_ids || []), ...(finding.evidence_ids || [])]) }));
}

// ---- agent run ------------------------------------------------------------------------
function scoreCards(box, score, steps, seconds) {
  clear(box);
  if (score) {
    box.append(
      card("Награда", (score.reward >= 0 ? "+" : "") + score.reward.toFixed(3), score.reward >= 0.5 ? "good" : score.reward < 0 ? "bad" : null),
      card("F1", score.f1.toFixed(3)),
      card("Точность P", score.precision.toFixed(3)),
      card("Полнота R", score.recall.toFixed(3)),
      card("Верно / ложно / пропуск", `${score.tp} / ${score.fp} / ${score.fn}`),
    );
  } else {
    box.append(card("Оценка", state.graded ? "у оценщика" : "нет эталона"));
  }
  if (steps !== undefined) box.append(card("Шагов", steps));
  if (seconds !== undefined) box.append(card("Время", `${(seconds * 1000).toFixed(0)} мс`));
}

async function runAgent() {
  if (!state.session) return;
  const policy = $("policy").value;
  await busy($("run-btn"), async () => {
    status(`Агент ${policy} проходит эпизод…`);
    const data = await api(`/api/sessions/${state.session}/run`, { policy });
    $("agent-empty").hidden = true;
    $("agent-result").hidden = false;
    scoreCards($("agent-cards"), data.score, data.steps, data.seconds);
    const list = clear($("agent-findings"));
    if (!data.findings.length) list.append(el("p", { class: "muted", text: "Агент не нашёл ошибок (пустой ответ)." }));
    for (const finding of data.findings) list.append(findingCard(finding));
    table($("agent-trajectory"), [
      { title: "Шаг", value: (t) => t.step },
      { title: "Действие", value: (t) => describeAction(t.action), wrap: true },
      { title: "Результат", value: (t) => describeObservation(t.observation), wrap: true },
      { title: "Награда", value: (t) => (t.reward === null || t.reward === undefined ? "—" : t.reward), num: true },
    ], data.trajectory);
    showTab("agent");
    const tail = data.score ? ` Награда ${data.score.reward.toFixed(3)}, F1 ${data.score.f1.toFixed(3)}.` : " Награда остаётся у оценщика.";
    status(`Агент ${policy}: ${data.findings.length} выводов за ${data.steps} шагов.` + tail);
  });
}

function describeAction(action) {
  if (!action || typeof action !== "object") return "—";
  switch (action.type) {
    case "reset": return "reset";
    case "list_records": return `list_records ${action.table}${action.cursor ? " · курсор" : ""}`;
    case "get_record": return `get_record ${action.table} ${action.id}`;
    case "get_document": return `get_document ${action.doc_id}`;
    case "calculator": return `calculator ${action.expression}`;
    case "submit": return `submit · выводов: ${((action.findings || {}).findings || []).length}`;
    default: return JSON.stringify(action).slice(0, 120);
  }
}

function describeObservation(obs) {
  if (!obs || typeof obs !== "object") return "—";
  if (obs.error) return `ошибка: ${obs.error}`;
  if (obs.status) return obs.status;
  if (obs.task) return "первое наблюдение: задача, manifest, политика";
  const result = obs.result;
  if (result && Array.isArray(result.records)) return `${result.records.length} записей${result.next_cursor ? ", есть следующая страница" : ""}`;
  if (result && typeof result === "object") return Object.keys(result).slice(0, 4).join(", ");
  return result === undefined ? "—" : String(result);
}

// ---- manual audit ----------------------------------------------------------------------
const ACTION_FIELDS = {
  list_records: [
    { name: "table", label: "Таблица", options: ["ledger", "invoices", "payments", "fx"] },
    { name: "limit", label: "Строк (1–50)", type: "number", value: 50 },
    { name: "cursor", label: "Курсор (пусто — с начала)", placeholder: "next_cursor из прошлого ответа" },
  ],
  get_record: [
    { name: "table", label: "Таблица", options: ["ledger", "invoices", "payments", "fx"] },
    { name: "id", label: "ID записи", placeholder: "row_… / invoice_… / payment_… / fx_…" },
  ],
  get_document: [{ name: "doc_id", label: "doc_id", placeholder: "doc_…" }],
  calculator: [{ name: "expression", label: "Выражение", placeholder: "(1500000 * 5.95) - 8925000" }],
};

function renderActionFields() {
  const box = clear($("action-fields"));
  for (const field of ACTION_FIELDS[$("action-type").value]) {
    const input = field.options
      ? el("select", { "data-field": field.name }, field.options.map((o) => el("option", { value: o, text: o })))
      : el("input", { "data-field": field.name, type: field.type || "text", value: field.value, placeholder: field.placeholder, autocomplete: "off" });
    box.append(el("label", {}, field.label, input));
  }
}

function resetManual() {
  $("steps-left").textContent = state.observation.remaining_steps ?? 100;
  $("observation").textContent = JSON.stringify(state.observation, null, 2).slice(0, 20000);
  clear($("history"));
  clear($("manual-result"));
  renderDrafts();
  setManualEnabled(true);
}

function setManualEnabled(enabled) {
  for (const id of ["action-btn", "submit-btn", "add-finding"]) $(id).disabled = !enabled;
}

function readAction() {
  const action = { type: $("action-type").value };
  for (const input of $("action-fields").querySelectorAll("[data-field]")) {
    const name = input.dataset.field;
    const raw = input.value.trim();
    if (name === "limit") action.limit = raw === "" ? null : Number.parseInt(raw, 10);
    else if (name === "cursor") action.cursor = raw === "" ? null : raw;
    else action[name] = raw;
  }
  return action;
}

function seenIds(result) {
  const ids = [];
  const take = (row) => {
    if (!row || typeof row !== "object") return;
    for (const key of ["doc_id", ...Object.values(TABLE_IDS)]) if (typeof row[key] === "string") ids.push(row[key]);
  };
  if (result && Array.isArray(result.records)) result.records.forEach(take);
  else take(result);
  return [...new Set(ids)];
}

async function doAction(event) {
  event.preventDefault();
  if (!state.session || state.finished) return;
  const action = readAction();
  await busy($("action-btn"), async () => {
    const data = await api(`/api/sessions/${state.session}/step`, { action });
    afterStep(action, data);
    if (action.type === "list_records" && data.observation.result && data.observation.result.next_cursor) {
      const cursor = $("action-fields").querySelector('[data-field="cursor"]');
      if (cursor) cursor.value = data.observation.result.next_cursor;
    }
  });
  setManualEnabled(!state.finished);
}

function afterStep(action, data) {
  const obs = data.observation;
  $("steps-left").textContent = obs.remaining_steps ?? "—";
  const pre = $("observation");
  pre.textContent = JSON.stringify(obs, null, 2).slice(0, 20000);
  const ids = seenIds(obs.result);
  const old = pre.previousElementSibling;
  if (old && old.classList.contains("obs-ids")) old.remove();
  if (ids.length) {
    const choice = (value, text, checked) => el("label", { class: "inline" },
      el("input", { type: "radio", name: "chip-target", value, checked }), text);
    pre.before(el("div", { class: "obs-ids" },
      el("span", { class: "hint", text: "Добавить в активный вывод как:" }),
      choice("evidence", "доказательство", true), choice("affected", "affected_id", false),
      ids.slice(0, 60).map((id) => el("button", { class: "chip", type: "button", text: "+ " + id, onclick: () => addEvidence(id) }))));
  }
  $("history").append(el("li", { class: obs.error ? "err" : null, text: `${describeAction(action)} → ${describeObservation(obs)}` }));
  if (data.terminated || data.truncated) finishManual(data);
}

function finishManual(data) {
  state.finished = true;
  setManualEnabled(false);
  const box = clear($("manual-result"));
  const cards = el("div", { class: "cards" });
  scoreCards(cards, data.score);
  box.append(el("h3", { text: data.terminated ? "Эпизод завершён" : "Эпизод прерван (лимит шагов или времени)" }), cards);
  if (!data.score && state.graded) box.append(el("p", { class: "hint", text: "В закрытом режиме награда и диагностика остаются у оценщика до конца прогона." }));
  status(data.score ? `Ваш аудит: награда ${data.score.reward.toFixed(3)}, F1 ${data.score.f1.toFixed(3)}.` : "Ответ отправлен.");
}

function newDraft() {
  return { type: "duplicate_invoice", affected_ids: [], claimed_amount_minor: "", currency: "KZT", evidence_ids: [], explanation: "" };
}

function addEvidence(id) {
  if (!state.drafts.length) { state.drafts.push(newDraft()); state.activeDraft = 0; }
  const draft = state.drafts[state.activeDraft];
  const target = document.querySelector('input[name="chip-target"]:checked');
  if (target && target.value === "affected") {
    if (!draft.affected_ids.includes(id)) draft.affected_ids.push(id);
  } else {
    if (draft.evidence_ids.length >= 10) { status("Не больше 10 доказательств на вывод", true); return; }
    if (!draft.evidence_ids.includes(id)) draft.evidence_ids.push(id);
  }
  renderDrafts();
}

function renderDrafts() {
  const box = clear($("drafts"));
  state.drafts.forEach((draft, index) => {
    const typeSelect = el("select", { onchange: (e) => { draft.type = e.target.value; } },
      Object.entries(TYPES).map(([k, t]) => el("option", { value: k, text: t, selected: draft.type === k })));
    const affected = el("input", { value: draft.affected_ids.join(", "), placeholder: "invoice_… или payment_…", oninput: (e) => {
      draft.affected_ids = e.target.value.split(/[\s,]+/).filter(Boolean);
    } });
    const amount = el("input", { type: "number", step: 1, value: draft.claimed_amount_minor, placeholder: "в минорных единицах", oninput: (e) => { draft.claimed_amount_minor = e.target.value; } });
    const currency = el("input", { value: draft.currency, maxlength: 3, oninput: (e) => { draft.currency = e.target.value.toUpperCase(); } });
    const explanation = el("input", { value: draft.explanation, maxlength: 1000, placeholder: "коротко, на оценку не влияет", oninput: (e) => { draft.explanation = e.target.value; } });
    const chips = el("div", { class: "ids" }, draft.evidence_ids.map((id) => el("button", { class: "chip", type: "button", title: "Убрать", onclick: () => {
      draft.evidence_ids = draft.evidence_ids.filter((x) => x !== id); renderDrafts();
    } }, id, el("span", { class: "x", text: "×" }))));
    box.append(el("div", { class: "draft" + (index === state.activeDraft ? " active" : ""), onclick: (e) => {
      if (state.activeDraft !== index && e.target.tagName !== "INPUT" && e.target.tagName !== "SELECT") { state.activeDraft = index; renderDrafts(); }
    } },
      el("label", {}, `Вывод ${index + 1}${index === state.activeDraft ? " · активный, «+» добавляет сюда" : ""}`, typeSelect),
      el("label", {}, "affected_ids (через запятую)", affected),
      el("div", { class: "row" }, el("label", {}, "Сумма", amount), el("label", {}, "Валюта", currency)),
      el("label", {}, "Объяснение", explanation),
      el("div", { class: "hint", text: `Доказательства (${draft.evidence_ids.length}/10):` }), chips,
      el("div", { class: "actions" }, el("span"),
        el("button", { class: "small", type: "button", text: "Удалить", onclick: (e) => {
          e.stopPropagation();
          state.drafts.splice(index, 1);
          state.activeDraft = Math.max(0, Math.min(state.activeDraft, state.drafts.length - 1));
          renderDrafts();
        } }))));
  });
  if (!state.drafts.length) box.append(el("p", { class: "muted", text: "Выводов нет: отправка пустого ответа означает «ошибок не найдено»." }));
}

async function submitFindings() {
  if (!state.session || state.finished) return;
  const findings = [];
  for (const [index, d] of state.drafts.entries()) {
    const amount = String(d.claimed_amount_minor).trim() === "" ? NaN : Number(d.claimed_amount_minor);
    if (!Number.isSafeInteger(amount)) { status(`Вывод ${index + 1}: сумма должна быть целым числом в минорных единицах`, true); return; }
    if (!d.affected_ids.length || !d.evidence_ids.length) { status(`Вывод ${index + 1}: нужны affected_ids и хотя бы одно доказательство`, true); return; }
    findings.push({ type: d.type, affected_ids: d.affected_ids, claimed_amount_minor: amount, currency: d.currency,
                    evidence_ids: d.evidence_ids, explanation: d.explanation });
  }
  if (!window.confirm(`Отправить ${findings.length} выводов? Эпизод завершится.`)) return;
  const action = { type: "submit", findings: { episode_id: state.files.manifest.episode_id, findings } };
  await busy($("submit-btn"), async () => {
    const data = await api(`/api/sessions/${state.session}/step`, { action });
    afterStep(action, data);
  });
  setManualEnabled(!state.finished);  // busy() re-enables the button it held
}

// ---- wiring --------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  buildErrorInputs();
  renderActionFields();
  $("preset").addEventListener("change", () => { $("custom-errors").hidden = $("preset").value !== "custom"; });
  $("create-form").addEventListener("submit", createEpisode);
  $("import-form").addEventListener("submit", importFolder);
  const applyTheme = (t) => { if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme; };
  try { applyTheme(localStorage.getItem("theme")); } catch (e) { /* storage unavailable */ }
  $("theme-btn").addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    applyTheme(next);
    try { localStorage.setItem("theme", next); } catch (e) { /* storage unavailable */ }
  });
  $("run-btn").addEventListener("click", runAgent);
  $("tabs").addEventListener("click", (e) => { if (e.target.dataset.tab) showTab(e.target.dataset.tab); });
  $("doc-search").addEventListener("input", renderDocuments);
  $("doc-kind").addEventListener("change", renderDocuments);
  $("ledger-search").addEventListener("input", () => { state.ledgerPage = 0; renderLedger(); });
  $("ledger-account").addEventListener("change", () => { state.ledgerPage = 0; renderLedger(); });
  $("fx-search").addEventListener("input", renderFx);
  $("action-type").addEventListener("change", renderActionFields);
  $("action-form").addEventListener("submit", doAction);
  $("add-finding").addEventListener("click", () => {
    if (state.drafts.length >= 30) { status("Не больше 30 выводов", true); return; }
    state.drafts.push(newDraft()); state.activeDraft = state.drafts.length - 1; renderDrafts();
  });
  $("submit-btn").addEventListener("click", submitFindings);
  try {
    const health = await api("/api/health");
    $("version").textContent = `RL-среда для ИИ-аудитора · v${health.version}`;
  } catch (error) {
    status("API недоступен: запустите python -m audit_env serve", true);
  }
});
