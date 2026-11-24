import { vi } from "vitest";

export class FakeXHR {
  static instances: FakeXHR[] = [];
  static latest(): FakeXHR {
    const xhr = FakeXHR.instances[FakeXHR.instances.length - 1];
    if (!xhr) throw new Error("No request was sent");
    return xhr;
  }

  upload: { onload: (() => void) | null } = { onload: null };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  ontimeout: (() => void) | null = null;
  onabort: (() => void) | null = null;
  status = 0;
  responseText = "";
  timeout = 0;
  method = "";
  url = "";
  headers: Record<string, string> = {};
  body: FormData | null = null;
  aborted = false;

  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }

  setRequestHeader(name: string, value: string) {
    this.headers[name] = value;
  }

  send(body: FormData) {
    this.body = body;
    FakeXHR.instances.push(this);
  }

  abort() {
    this.aborted = true;
    this.onabort?.();
  }

  finishUpload() {
    this.upload.onload?.();
  }

  respond(status: number, body: unknown) {
    this.status = status;
    this.responseText = typeof body === "string" ? body : JSON.stringify(body);
    this.onload?.();
  }
}

export function installFakeXhr() {
  FakeXHR.instances = [];
  vi.stubGlobal("XMLHttpRequest", FakeXHR);
}
