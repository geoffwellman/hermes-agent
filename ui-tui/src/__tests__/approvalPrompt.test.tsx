import { renderSync } from '@hermes/ink'
import React from 'react'
import { Writable } from 'stream'
import { afterEach, describe, expect, it } from 'vitest'

import { ApprovalPrompt } from '../components/prompts.js'
import { DEFAULT_THEME } from '../theme.js'

class CaptureStream extends Writable {
  columns = 120
  isTTY = true
  rows = 40
  chunks: string[] = []

  _write(chunk: unknown, _encoding: BufferEncoding, callback: (error?: Error | null) => void) {
    this.chunks.push(Buffer.isBuffer(chunk) ? chunk.toString('utf8') : String(chunk))
    callback()
  }
}

describe('ApprovalPrompt', () => {
  const instances: Array<{ cleanup: () => void; unmount: () => void }> = []

  afterEach(() => {
    for (const instance of instances.splice(0)) {
      instance.unmount()
      instance.cleanup()
    }
  })

  it('emits a terminal bell when shown', async () => {
    const stdout = new CaptureStream() as CaptureStream & NodeJS.WriteStream
    const stderr = new CaptureStream() as CaptureStream & NodeJS.WriteStream
    const stdin = process.stdin

    const instance = renderSync(
      React.createElement(ApprovalPrompt, {
        onChoice: () => undefined,
        req: { command: 'rm -rf /tmp/example', description: 'Dangerous command' },
        t: DEFAULT_THEME
      }),
      { exitOnCtrlC: false, patchConsole: false, stderr, stdin, stdout }
    )

    instances.push(instance)
    await new Promise(resolve => setTimeout(resolve, 0))

    expect(stdout.chunks.join('')).toContain('\u0007')
  })
})
