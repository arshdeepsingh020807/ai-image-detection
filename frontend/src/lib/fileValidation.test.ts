import { describe, expect, it } from "vitest";
import { asciiBytes, JPEG_BYTES, makeFile, pngBytes, sizedFile, webpBytes } from "../test/files";
import { checkDimensions, isAnimated, MAX_IMAGE_PIXELS, MAX_UPLOAD_BYTES, sniffFormat, validateImageFile } from "./fileValidation";

const bytes = (list: number[]) => new Uint8Array(list);

describe("sniffFormat", () => {
  it.each([
    ["JPEG", JPEG_BYTES],
    ["PNG", pngBytes()],
    ["WEBP", webpBytes("VP8 ")],
    ["GIF", asciiBytes("GIF89a....")],
    ["BMP", asciiBytes("BM......")],
    ["TIFF", [...asciiBytes("II*"), 0, 8, 0, 0, 0]],
    ["TIFF", [...asciiBytes("MM"), 0, 42, 0, 0, 0, 8]],
    ["HEIC", [0, 0, 0, 24, ...asciiBytes("ftypheic")]],
    ["HEIC", [0, 0, 0, 24, ...asciiBytes("ftypmif1")]],
    ["AVIF", [0, 0, 0, 24, ...asciiBytes("ftypavif")]],
  ])("recognises %s", (format, data) => {
    expect(sniffFormat(bytes(data))).toBe(format);
  });

  it.each([
    ["text", asciiBytes("hello world")],
    ["RIFF that is not WebP", asciiBytes("RIFF\0\0\0\0WAVEfmt ")],
    ["MP4 container", [0, 0, 0, 24, ...asciiBytes("ftypisom")]],
    ["a truncated JPEG marker", [0xff, 0xd8]],
    ["nothing", []],
  ])("returns null for %s", (_, data) => {
    expect(sniffFormat(bytes(data))).toBeNull();
  });
});

describe("isAnimated", () => {
  it("detects an APNG acTL chunk before the image data", () => {
    expect(isAnimated(bytes(pngBytes(true)), "PNG")).toBe(true);
    expect(isAnimated(bytes(pngBytes(false)), "PNG")).toBe(false);
  });

  it("stops at a truncated PNG header", () => {
    expect(isAnimated(bytes(pngBytes().slice(0, 12)), "PNG")).toBe(false);
  });

  it("reads the WebP VP8X animation flag", () => {
    expect(isAnimated(bytes(webpBytes("VP8X", 0x02)), "WEBP")).toBe(true);
    expect(isAnimated(bytes(webpBytes("VP8X", 0x10)), "WEBP")).toBe(false);
    expect(isAnimated(bytes(webpBytes("VP8 ")), "WEBP")).toBe(false);
  });

  it("treats JPEG as still", () => {
    expect(isAnimated(bytes(JPEG_BYTES), "JPEG")).toBe(false);
  });
});

describe("validateImageFile", () => {
  const rejected = async (file: File) => {
    const result = await validateImageFile(file);
    if (result.ok) throw new Error("expected rejection");
    return result.error;
  };

  it.each([
    ["JPEG", makeFile(JPEG_BYTES, "a.jpg")],
    ["PNG", makeFile(pngBytes(), "a.png", "image/png")],
    ["WEBP", makeFile(webpBytes("VP8 "), "a.webp", "image/webp")],
    ["JPEG", makeFile(JPEG_BYTES, "misnamed.png", "image/png")],
  ])("accepts %s by content", async (format, file) => {
    expect(await validateImageFile(file)).toEqual({ ok: true, format });
  });

  it("rejects an empty file", async () => {
    expect((await rejected(makeFile([], "empty.jpg"))).code).toBe("empty_file");
  });

  it("rejects files over 10 MB and states the size", async () => {
    const error = await rejected(sizedFile(MAX_UPLOAD_BYTES + 1));
    expect(error.code).toBe("file_too_large");
    expect(error.detail).toContain("10.0 MB");
  });

  it("accepts a file of exactly 10 MB", async () => {
    expect((await validateImageFile(sizedFile(MAX_UPLOAD_BYTES))).ok).toBe(true);
  });

  it("rejects text renamed to .jpg as an invalid image and says why", async () => {
    const error = await rejected(makeFile(asciiBytes("not really a jpeg"), "photo.JPG"));
    expect(error.code).toBe("invalid_image");
    expect(error.detail).toContain("“.jpg”");
  });

  it("rejects unknown content with another extension as unsupported", async () => {
    expect((await rejected(makeFile(asciiBytes("plain text"), "notes.txt", "text/plain"))).code).toBe("unsupported_format");
    expect((await rejected(makeFile(asciiBytes("plain text"), "README", ""))).code).toBe("unsupported_format");
  });

  it.each([
    ["HEIC", [0, 0, 0, 24, ...asciiBytes("ftypheic")], /HEIC.*JPEG/],
    ["GIF", asciiBytes("GIF89a"), /GIF/],
    ["BMP", asciiBytes("BM......"), /This is a BMP image/],
  ])("names a recognised but unsupported %s file", async (_, data, message) => {
    const error = await rejected(makeFile(data, "file.bin"));
    expect(error.code).toBe("unsupported_format");
    expect(error.detail).toMatch(message);
  });

  it("rejects animated PNG and WebP", async () => {
    expect((await rejected(makeFile(pngBytes(true), "anim.png"))).code).toBe("animated_image");
    expect((await rejected(makeFile(webpBytes("VP8X", 0x02), "anim.webp"))).code).toBe("animated_image");
  });
});

describe("checkDimensions", () => {
  it("allows up to 40 megapixels", () => {
    expect(checkDimensions(8000, 5000)).toBeNull();
    expect(8000 * 5000).toBe(MAX_IMAGE_PIXELS);
  });

  it("rejects more than 40 megapixels with the measured size", () => {
    const error = checkDimensions(8001, 5000);
    expect(error?.code).toBe("image_too_large");
    expect(error?.detail).toContain("8001 × 5000 (40.0 megapixels)");
  });
});
