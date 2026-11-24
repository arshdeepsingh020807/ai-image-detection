export const JPEG_BYTES = [0xff, 0xd8, 0xff, 0xe0, 0, 0x10, 0x4a, 0x46, 0x49, 0x46];
export const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

const ascii = (text: string) => Array.from({ length: text.length }, (_, i) => text.charCodeAt(i));
const u32 = (n: number) => [(n >>> 24) & 255, (n >>> 16) & 255, (n >>> 8) & 255, n & 255];
const chunk = (type: string, length: number) => [...u32(length), ...ascii(type), ...new Array<number>(length + 4).fill(0)];

export const pngBytes = (animated = false) => [
  ...PNG_SIGNATURE,
  ...chunk("IHDR", 13),
  ...(animated ? chunk("acTL", 8) : []),
  ...chunk("IDAT", 4),
];

export const webpBytes = (kind: "VP8 " | "VP8X", flags = 0) => [
  ...ascii("RIFF"), 0, 0, 0, 0, ...ascii("WEBP"), ...ascii(kind), 10, 0, 0, 0, flags, 0, 0, 0,
];

export const asciiBytes = ascii;

export const makeFile = (bytes: number[] | Uint8Array, name = "image.jpg", type = "image/jpeg") =>
  new File([new Uint8Array(bytes)], name, { type });

export function sizedFile(size: number, name = "big.jpg") {
  const file = makeFile(JPEG_BYTES, name);
  Object.defineProperty(file, "size", { value: size });
  return file;
}
