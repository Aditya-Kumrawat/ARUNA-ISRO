import initSqlJs, { type Database as SqlJsDatabase, type SqlJsStatic } from 'sql.js'
import { readFileSync, existsSync, statSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import type { FlareEvent } from './types.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PROJECT_ROOT = path.resolve(__dirname, '..', '..')
export const DB_PATH = process.env.ARUNA_DB_PATH ?? path.join(PROJECT_ROOT, 'aruna_catalog.db')

let _SQL: SqlJsStatic | null = null
let _db: SqlJsDatabase | null = null
let _dbMtime = 0

export async function initDatabase(): Promise<void> {
  _SQL = await initSqlJs()
  await loadDb()
}

async function loadDb(): Promise<void> {
  if (!_SQL) return
  if (!existsSync(DB_PATH)) return
  try {
    const mtime = statSync(DB_PATH).mtimeMs
    if (mtime === _dbMtime) return
    const buf = readFileSync(DB_PATH)
    _db?.close()
    _db = new _SQL.Database(buf)
    _dbMtime = mtime
  } catch {
    // DB may be mid-write; skip this cycle
  }
}

// Called by poller every 5 s to detect new Python writes
export async function refreshIfNeeded(): Promise<void> {
  await loadDb()
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type BindParam = any

function execToObjects(sql: string, params?: BindParam[]): Record<string, unknown>[] {
  if (!_db) return []
  try {
    const results = _db.exec(sql, params)
    if (!results.length) return []
    const { columns, values } = results[0]
    return values.map(row => {
      const obj: Record<string, unknown> = {}
      columns.forEach((col, i) => { obj[col] = row[i] })
      return obj
    })
  } catch {
    return []
  }
}

function execGetOne(sql: string, params?: BindParam[]): Record<string, unknown> | undefined {
  const rows = execToObjects(sql, params)
  return rows[0]
}

function execScalar<T>(sql: string, params?: BindParam[], col = 0): T | undefined {
  if (!_db) return undefined
  try {
    const r = _db.exec(sql, params)
    return r[0]?.values[0]?.[col] as T ?? undefined
  } catch {
    return undefined
  }
}

function rowToEvent(r: Record<string, unknown>): FlareEvent {
  return {
    event_id: String(r.event_id ?? ''),
    start_time: (r.start_time as string) ?? null,
    hope_time: (r.hope_time as string) ?? null,
    onset_time: (r.onset_time as string) ?? null,
    peak_time: (r.peak_time as string) ?? null,
    end_time: (r.end_time as string) ?? null,
    goes_class: String(r.goes_class ?? 'B'),
    goes_subclass: Number(r.goes_subclass ?? 0),
    peak_flux_solexs: Number(r.peak_flux_solexs ?? 0),
    peak_flux_hel1os: Number(r.peak_flux_hel1os ?? 0),
    peak_T_MK: Number(r.peak_T_MK ?? 0),
    peak_EM_cm3: Number(r.peak_EM_cm3 ?? 0),
    peak_DNI: Number(r.peak_DNI ?? 0),
    hope_lead_time_sec: r.hope_lead_time_sec != null ? Number(r.hope_lead_time_sec) : null,
    hope_fired: Boolean(r.hope_fired),
    instruments: tryJson<string[]>(r.instruments as string, []),
    detection_method: String(r.detection_method ?? ''),
    confidence: Number(r.confidence ?? 0),
    flare_type: String(r.flare_type ?? ''),
    pipeline_version: String(r.pipeline_version ?? ''),
    provenance_hash: String(r.provenance_hash ?? ''),
  }
}

function tryJson<T>(s: string | null | undefined, fallback: T): T {
  if (!s) return fallback
  try { return JSON.parse(s) as T } catch { return fallback }
}

export interface ListEventsOpts {
  limit?: number
  offset?: number
  goesClass?: string
  dateFrom?: string
  dateTo?: string
}

export function listEvents(opts: ListEventsOpts = {}): FlareEvent[] {
  const { limit = 20, offset = 0, goesClass, dateFrom, dateTo } = opts
  const clauses: string[] = []
  const params: (string | number | null)[] = []
  if (goesClass) { clauses.push('goes_class = ?'); params.push(goesClass) }
  if (dateFrom) { clauses.push('peak_time >= ?'); params.push(dateFrom) }
  if (dateTo) { clauses.push('peak_time <= ?'); params.push(dateTo) }
  const where = clauses.length ? `WHERE ${clauses.join(' AND ')}` : ''
  return execToObjects(
    `SELECT * FROM master_flare_catalog ${where} ORDER BY peak_time DESC LIMIT ${limit} OFFSET ${offset}`,
    params
  ).map(rowToEvent)
}

export function getEvent(id: string): FlareEvent | null {
  const row = execGetOne('SELECT * FROM master_flare_catalog WHERE event_id = ?', [id])
  return row ? rowToEvent(row) : null
}

export function countEvents(opts: Omit<ListEventsOpts, 'limit' | 'offset'> = {}): number {
  const { goesClass, dateFrom, dateTo } = opts
  const clauses: string[] = []
  const params: (string | number | null)[] = []
  if (goesClass) { clauses.push('goes_class = ?'); params.push(goesClass) }
  if (dateFrom) { clauses.push('peak_time >= ?'); params.push(dateFrom) }
  if (dateTo) { clauses.push('peak_time <= ?'); params.push(dateTo) }
  const where = clauses.length ? `WHERE ${clauses.join(' AND ')}` : ''
  return execScalar<number>(`SELECT COUNT(*) FROM master_flare_catalog ${where}`, params) ?? 0
}

export function getRecentEvents(limit = 10): FlareEvent[] {
  return execToObjects(
    `SELECT * FROM master_flare_catalog ORDER BY rowid DESC LIMIT ${limit}`,
    []
  ).map(rowToEvent)
}

export function getNewEventsSince(lastRowid: number): { events: FlareEvent[]; maxRowid: number } {
  const rows = execToObjects(
    `SELECT rowid, * FROM master_flare_catalog WHERE rowid > ${lastRowid} ORDER BY rowid ASC`,
    []
  )
  const events = rows.map(rowToEvent)
  const maxRowid = rows.length > 0 ? Number(rows[rows.length - 1].rowid) : lastRowid
  return { events, maxRowid }
}

export function getMaxRowid(): number {
  return execScalar<number>('SELECT MAX(rowid) FROM master_flare_catalog', []) ?? 0
}

export interface ValidationStats {
  total_events: number
  hope_fired_count: number
  hope_accuracy_pct: number
  dual_trigger_count: number
  class_distribution: Record<string, number>
  avg_lead_time_sec: number | null
  avg_confidence: number
}

export function getValidationStats(): ValidationStats {
  const total = execScalar<number>('SELECT COUNT(*) FROM master_flare_catalog') ?? 0
  const hope_fired = execScalar<number>('SELECT COUNT(*) FROM master_flare_catalog WHERE hope_fired = 1') ?? 0
  const dual = execScalar<number>("SELECT COUNT(*) FROM master_flare_catalog WHERE detection_method LIKE '%DUAL%'") ?? 0
  const avgLead = execScalar<number>('SELECT AVG(hope_lead_time_sec) FROM master_flare_catalog WHERE hope_lead_time_sec IS NOT NULL')
  const avgConf = execScalar<number>('SELECT AVG(confidence) FROM master_flare_catalog') ?? 0

  const classRows = execToObjects('SELECT goes_class, COUNT(*) as n FROM master_flare_catalog GROUP BY goes_class')
  const class_distribution: Record<string, number> = {}
  for (const r of classRows) class_distribution[String(r.goes_class)] = Number(r.n)

  return {
    total_events: Number(total),
    hope_fired_count: Number(hope_fired),
    hope_accuracy_pct: total > 0 ? Math.round((Number(hope_fired) / Number(total)) * 100) : 0,
    dual_trigger_count: Number(dual),
    class_distribution,
    avg_lead_time_sec: avgLead != null ? Math.round(Number(avgLead)) : null,
    avg_confidence: Number(avgConf),
  }
}
