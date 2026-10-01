import { afterEach, describe, expect, test } from "bun:test"
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"

const roots: string[] = []
const extensionPath = join(import.meta.dir, "../../../../omo-senpi/plugin/extensions/omo.js")
const senpiEntry = join(dirname(fileURLToPath(import.meta.resolve("@code-yeongyu/senpi"))), "bundle/cli.js")

type RecordedRequest = {
  readonly tools: readonly string[]
  readonly lastUser: string
}

type Deferred<T> = {
  readonly promise: Promise<T>
  readonly resolve: (value: T) => void
  readonly reject: (error: unknown) => void
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void
  let reject!: (error: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

function sse(chunks: readonly unknown[]): Response {
  const body = `${chunks.map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`).join("")}data: [DONE]\n\n`
  return new Response(body, { headers: { "content-type": "text/event-stream" } })
}

const base = { id: "mock", object: "chat.completion.chunk", created: 0, model: "mock-model" }

function text(content: string): Response {
  return sse([
    { ...base, choices: [{ index: 0, delta: { role: "assistant", content }, finish_reason: null }] },
    { ...base, choices: [{ index: 0, delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } },
  ])
}

function toolCall(name: string, args: unknown): Response {
  return sse([
    { ...base, choices: [{ index: 0, delta: { role: "assistant", tool_calls: [{ index: 0, id: "call_task", type: "function", function: { name, arguments: JSON.stringify(args) } }] }, finish_reason: null }] },
    { ...base, choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } },
  ])
}

async function withTimeout<T>(promise: Promise<T>, label: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<never>((_, reject) => {
    timer = setTimeout(() => reject(new Error(`timed out waiting for ${label}`)), 20_000)
  })
  try {
    return await Promise.race([promise, timeout])
  } finally {
    if (timer !== undefined) clearTimeout(timer)
  }
}

async function recordChildTools(mode: "in-process" | "process"): Promise<readonly string[]> {
  const root = mkdtempSync(join(tmpdir(), `omo-child-tools-${mode}-`))
  roots.push(root)
  const home = join(root, "home")
  const agentDir = join(home, ".omo/agent")
  const work = join(root, "work")
  const sessionTmp = join(root, "tmp")
  mkdirSync(agentDir, { recursive: true })
  mkdirSync(work, { recursive: true })
  mkdirSync(sessionTmp, { recursive: true })
  const git = Bun.spawnSync(["git", "init", "-q", work])
  if (git.exitCode !== 0) throw new Error(git.stderr.toString())

  const childRequest = deferred<RecordedRequest>()
  let parentTurns = 0
  const server = Bun.serve({
    port: 0,
    async fetch(request) {
      const url = new URL(request.url)
      if (!url.pathname.endsWith("/chat/completions")) return Response.json({})
      const body = await request.json() as {
        readonly tools?: ReadonlyArray<{ readonly function?: { readonly name?: string } }>
        readonly messages?: ReadonlyArray<{ readonly role: string; readonly content: unknown }>
      }
      const tools = (body.tools ?? []).map((entry) => entry.function?.name ?? "?").sort()
      const users = (body.messages ?? []).filter((message) => message.role === "user")
      const lastUser = JSON.stringify(users.at(-1)?.content ?? "")
      const isParent = tools.includes("task") && JSON.stringify(body.messages ?? []).includes("PARENT_PROBE")
      if (lastUser.includes("CHILD_PROBE")) childRequest.resolve({ tools, lastUser })
      if (isParent && parentTurns++ === 0) {
        return toolCall("task", {
          category: "unspecified-low",
          description: "builtin tool parity probe",
          prompt: "CHILD_PROBE: report done.",
          run_in_background: false,
          load_skills: [],
        })
      }
      return text(isParent ? "parent done" : "child done")
    },
  })

  const models = {
    providers: {
      mock: {
        baseUrl: `http://127.0.0.1:${server.port}/v1`,
        apiKey: "dummy",
        api: "openai-completions",
        models: [{ id: "mock-model", name: "Mock", api: "openai-completions", contextWindow: 128_000, maxTokens: 4096, input: ["text"] }],
      },
    },
  }
  const config = {
    categories: { "unspecified-low": { models: ["mock/mock-model"] } },
    task: mode === "process"
      ? { default_execution_mode: "process", process_runner: "child-process" }
      : { default_execution_mode: "in-process" },
  }
  writeFileSync(join(agentDir, "models.json"), JSON.stringify(models))
  writeFileSync(join(agentDir, "omo.json"), JSON.stringify(config))
  mkdirSync(join(home, ".omo"), { recursive: true })
  writeFileSync(join(home, ".omo/omo.jsonc"), JSON.stringify(config))

  const processResult = Bun.spawn(
    [process.execPath, senpiEntry, "--provider", "mock", "--model", "mock-model", "--no-ask-user", "--extension", extensionPath, "-p", "PARENT_PROBE: delegate one task."],
    {
      cwd: work,
      env: {
        HOME: home,
        PATH: process.env.PATH ?? "",
        TERM: "dumb",
        TMPDIR: sessionTmp,
        USER: "probe",
        LANG: "en_US.UTF-8",
        SENPI_CODING_AGENT_DIR: agentDir,
      },
      stdout: "pipe",
      stderr: "pipe",
    },
  )

  try {
    const recorded = await withTimeout(childRequest.promise, `${mode} child request`)
    return recorded.tools
  } finally {
    server.stop(true)
    processResult.kill()
    await processResult.exited
  }
}

afterEach(() => {
  while (roots.length > 0) rmSync(roots.pop() ?? "", { recursive: true, force: true })
})

describe("in-process child builtin tool parity", () => {
  test("#given matching category defaults #when each runner spawns a child #then tool names and counts match with web_search available", async () => {
    const inProcess = await recordChildTools("in-process")
    const processChild = await recordChildTools("process")

    expect(inProcess).toContain("web_search")
    expect(inProcess).toHaveLength(processChild.length)
    expect(inProcess).toEqual(processChild)
  }, 60_000)
})
