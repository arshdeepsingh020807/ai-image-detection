import { ApiError } from "./client";

export interface FriendlyError {
  code: string;
  title: string;
  detail: string;
  retryable: boolean;
}

type Message = Omit<FriendlyError, "code">;

const MESSAGES: Record<string, Message> = {
  missing_file: { title: "No image was received", detail: "The upload arrived without a file. Choose an image and try again.", retryable: true },
  empty_file: { title: "The file is empty", detail: "It contains no data. Choose a different image.", retryable: false },
  invalid_image: { title: "This file could not be read as an image", detail: "It may be damaged, truncated or not an image at all. Try exporting it again as JPEG or PNG.", retryable: false },
  unsupported_format: { title: "Unsupported file type", detail: "Only JPEG, PNG and WebP images can be analysed.", retryable: false },
  animated_image: { title: "Animated images are not supported", detail: "Export a single still frame as JPEG or PNG and upload that instead.", retryable: false },
  image_too_large: { title: "Image dimensions are too large", detail: "The image has more than 40 megapixels. Resize it and try again.", retryable: false },
  file_too_large: { title: "File is larger than 10 MB", detail: "Compress or resize the image so the file is 10 MB or smaller.", retryable: false },
  invalid_request: { title: "The server could not read the upload", detail: "The request was incomplete or malformed, for example because the connection dropped mid-upload. Try again.", retryable: true },
  rate_limited: { title: "Too many requests", detail: "The service limits how often images can be analysed. Wait a minute, then try again.", retryable: true },
  model_unavailable: { title: "The detection model is unavailable", detail: "The service is running but its model is not loaded. Try again shortly.", retryable: true },
  not_found: { title: "Analysis service not found", detail: "The API endpoint did not respond as expected. The deployment may be incomplete.", retryable: false },
  method_not_allowed: { title: "The service rejected this request", detail: "The API does not accept this kind of request at this address. The deployment may be misconfigured.", retryable: false },
  internal_error: { title: "The server hit an unexpected error", detail: "The server failed while processing the request. Try again; if it keeps happening, try a different image.", retryable: true },
  network_error: { title: "Could not reach the server", detail: "Check your connection and that the service is running, then try again.", retryable: true },
  timeout: { title: "The request took too long", detail: "The server did not answer in time, so the request was stopped. Try again.", retryable: true },
  bad_response: { title: "Unexpected response from the server", detail: "The service answered in a format this page does not understand.", retryable: true },
};

export function toFriendlyError(error: unknown): FriendlyError {
  const apiError = error instanceof ApiError ? error : new ApiError("unexpected", error instanceof Error ? error.message : "Request failed.");
  const known = MESSAGES[apiError.code];
  if (known) return { code: apiError.code, ...known };
  return { code: apiError.code, title: "The request failed", detail: apiError.message, retryable: true };
}

export const friendlyMessage = (code: string, detail?: string): FriendlyError => {
  const friendly = toFriendlyError(new ApiError(code, detail ?? ""));
  return detail ? { ...friendly, detail } : friendly;
};
