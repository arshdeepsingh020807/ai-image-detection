import { friendlyMessage, type FriendlyError } from "../api/errors";
import type { ImageFormat } from "../api/types";
import { formatBytes } from "./format";

export const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
export const MAX_IMAGE_PIXELS = 40_000_000;
export const ACCEPT = "image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp";
const HEADER_BYTES = 64 * 1024;
const ACCEPTED: readonly string[] = ["JPEG", "PNG", "WEBP"] satisfies ImageFormat[];
const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const HEIF_BRANDS = ["heic", "heix", "hevc", "heim", "heis", "mif1", "msf1"];

export type SniffedFormat = ImageFormat | "GIF" | "BMP" | "TIFF" | "HEIC" | "AVIF";
export type ValidationResult = { ok: true; format: ImageFormat } | { ok: false; error: FriendlyError };

const ascii = (bytes: Uint8Array, offset: number, length: number) =>
  String.fromCharCode(...bytes.subarray(offset, offset + length));
const startsWith = (bytes: Uint8Array, signature: number[]) => signature.every((byte, i) => bytes[i] === byte);

export function sniffFormat(bytes: Uint8Array): SniffedFormat | null {
  if (startsWith(bytes, [0xff, 0xd8, 0xff])) return "JPEG";
  if (startsWith(bytes, PNG_SIGNATURE)) return "PNG";
  if (ascii(bytes, 0, 4) === "RIFF" && ascii(bytes, 8, 4) === "WEBP") return "WEBP";
  if (ascii(bytes, 0, 4) === "GIF8") return "GIF";
  if (ascii(bytes, 0, 2) === "BM") return "BMP";
  if (["II*\0", "MM\0*"].includes(ascii(bytes, 0, 4))) return "TIFF";
  if (ascii(bytes, 4, 4) === "ftyp") {
    const brand = ascii(bytes, 8, 4);
    if (brand === "avif" || brand === "avis") return "AVIF";
    if (HEIF_BRANDS.includes(brand)) return "HEIC";
  }
  return null;
}

export function isAnimated(bytes: Uint8Array, format: ImageFormat): boolean {
  if (format === "WEBP") return ascii(bytes, 12, 4) === "VP8X" && ((bytes[20] ?? 0) & 0x02) !== 0;
  if (format !== "PNG") return false;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let offset = PNG_SIGNATURE.length; offset + 8 <= bytes.length; offset += 12 + view.getUint32(offset)) {
    const type = ascii(bytes, offset + 4, 4);
    if (type === "acTL") return true;
    if (type === "IDAT") return false;
  }
  return false;
}

const extensionOf = (name: string) => /\.([a-z0-9]+)$/i.exec(name)?.[1]?.toLowerCase() ?? "";

function rejectUnknown(name: string): FriendlyError {
  const extension = extensionOf(name);
  return ["jpg", "jpeg", "png", "webp"].includes(extension)
    ? friendlyMessage("invalid_image", `The name ends in “.${extension}”, but the contents are not a JPEG, PNG or WebP image. The file may be damaged or renamed.`)
    : friendlyMessage("unsupported_format", "The contents do not match any supported image type. Only JPEG, PNG and WebP images can be analysed.");
}

const REJECTED_HINTS: Partial<Record<SniffedFormat, string>> = {
  HEIC: "This is a HEIC image, the default for many phone cameras. Export or share it as JPEG, then upload that.",
  GIF: "This is a GIF image. Convert a still frame to PNG or JPEG.",
};

export async function validateImageFile(file: File): Promise<ValidationResult> {
  if (file.size === 0) return { ok: false, error: friendlyMessage("empty_file") };
  if (file.size > MAX_UPLOAD_BYTES) {
    return { ok: false, error: friendlyMessage("file_too_large", `This file is ${formatBytes(file.size)}; the limit is 10 MB.`) };
  }
  const bytes = new Uint8Array(await file.slice(0, HEADER_BYTES).arrayBuffer());
  const format = sniffFormat(bytes);
  if (format === null) return { ok: false, error: rejectUnknown(file.name) };
  if (!ACCEPTED.includes(format)) {
    const hint = REJECTED_HINTS[format] ?? `This is a ${format} image. Only JPEG, PNG and WebP images can be analysed.`;
    return { ok: false, error: friendlyMessage("unsupported_format", hint) };
  }
  const accepted = format as ImageFormat;
  if (isAnimated(bytes, accepted)) return { ok: false, error: friendlyMessage("animated_image") };
  return { ok: true, format: accepted };
}

export function checkDimensions(width: number, height: number): FriendlyError | null {
  const pixels = width * height;
  return pixels > MAX_IMAGE_PIXELS
    ? friendlyMessage("image_too_large", `This image is ${width} × ${height} (${(pixels / 1e6).toFixed(1)} megapixels); the limit is 40 megapixels.`)
    : null;
}
