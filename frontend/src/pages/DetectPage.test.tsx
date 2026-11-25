import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { FakeXHR, installFakeXhr } from "../test/fakeXhr";
import { mockFetch, ok, type Route } from "../test/fetchMock";
import { fixtures } from "../test/fixtures";
import { asciiBytes, JPEG_BYTES, makeFile } from "../test/files";
import { loadImage } from "../test/render";
import { DetectPage } from "./DetectPage";

const photo = () => makeFile(JPEG_BYTES, "photo.jpg");

function setup(model: Route = ok(fixtures.model())) {
  installFakeXhr();
  mockFetch({ "/api/model": model });
  const user = userEvent.setup();
  render(<DetectPage />);
  return user;
}

async function selectPhoto(user: ReturnType<typeof userEvent.setup>, file = photo(), size: [number, number] = [1024, 512]) {
  await user.upload(screen.getByLabelText(/drop an image here/i), file);
  const preview = await screen.findByAltText(`Preview of ${file.name}`);
  loadImage(preview, ...size);
  return preview;
}

async function analyse(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: "Analyse image" }));
  return FakeXHR.latest();
}

async function respond(xhr: FakeXHR, status: number, body: unknown) {
  await act(async () => {
    xhr.finishUpload();
    xhr.respond(status, body);
    await Promise.resolve();
  });
}

const fire = (event: () => void) => act(async () => {
  event();
  await Promise.resolve();
});

describe("DetectPage", () => {
  beforeEach(() => { document.title = ""; });

  it("starts idle with a drop zone, an empty result panel and the limitations note", async () => {
    setup();
    expect(screen.getByRole("heading", { level: 1, name: "Detect AI-generated images" })).toBeInTheDocument();
    expect(screen.getByLabelText(/drop an image here/i)).toHaveAttribute("type", "file");
    expect(screen.getByText(/the verdict, calibrated probabilities and heatmap will appear here/i)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "How to read a result" })).toBeInTheDocument();
    expect(screen.getByText(/a statistical estimate, not proof/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Analyse image" })).not.toBeInTheDocument();
    await waitFor(() => { expect(document.title).toBe("Detect AI-generated images · AI Image Detection"); });
    expect(screen.queryByText(/untrained stand-in/i)).not.toBeInTheDocument();
  });

  it("warns up front when the deployed model is untrained", async () => {
    setup(ok({ ...fixtures.model(), trained: false, version: "untrained-standin" }));
    expect(await screen.findByRole("heading", { name: "The deployed model is an untrained stand-in" })).toBeInTheDocument();
    expect(screen.getByText("untrained-standin")).toBeInTheDocument();
  });

  it("shows the preview and announces the selection", async () => {
    const user = setup();
    await selectPhoto(user);
    expect(screen.getByRole("status")).toHaveTextContent("Image selected: photo.jpg. Ready to analyse.");
    expect(screen.getByText("photo.jpg")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /include heatmap/i })).toBeChecked();
    expect(screen.getByText(/ready\. select “analyse image”/i)).toBeInTheDocument();
  });

  it("accepts a dropped file", async () => {
    setup();
    fireEvent.drop(screen.getByLabelText(/drop an image here/i).closest(".dropzone") as Element, { dataTransfer: { files: [photo()] } });
    expect(await screen.findByAltText("Preview of photo.jpg")).toBeInTheDocument();
  });

  it("rejects a renamed text file before uploading and focuses the message", async () => {
    const user = setup();
    await user.upload(screen.getByLabelText(/drop an image here/i), makeFile(asciiBytes("hello"), "fake.jpg"));
    const alert = await screen.findByRole("alert");
    expect(within(alert).getByRole("heading")).toHaveTextContent("This file could not be read as an image");
    expect(within(alert).getByRole("heading")).toHaveFocus();
    expect(FakeXHR.instances).toHaveLength(0);
  });

  it("walks through uploading, analysing and an AI-generated result with the heatmap on the analysed region", async () => {
    const user = setup();
    await selectPhoto(user);
    const xhr = await analyse(user);
    expect(xhr.body?.get("explain")).toBe("true");
    expect(screen.getByRole("status")).toHaveTextContent("Uploading image…");
    expect(screen.getByText("Upload").closest("li")).toHaveAttribute("aria-current", "step");
    expect(screen.getByRole("button", { name: "Analyse image" })).toBeDisabled();

    act(() => { xhr.finishUpload(); });
    expect(screen.getByRole("status")).toHaveTextContent("Upload complete. Running the model…");
    expect(screen.getByText("Analyse").closest("li")).toHaveAttribute("aria-current", "step");

    await fire(() => { xhr.respond(200, fixtures.prediction.ai()); });
    const heading = screen.getByRole("heading", { name: "Likely AI-generated" });
    expect(heading).toHaveFocus();
    expect(screen.getByRole("status")).toHaveTextContent("Analysis complete. Verdict: Likely AI-generated.");
    expect(screen.getAllByText("93.0%").length).toBeGreaterThan(0);
    expect(screen.getByText((_, el) => el?.tagName === "CODE" && el.textContent === "fixture-v1")).toBeInTheDocument();
    expect(screen.getByText(/^Brighter, more opaque areas influenced the AI-generated score more\./)).toBeInTheDocument();

    const region = screen.getByTestId("heatmap-region");
    expect(parseFloat(region.style.left)).toBeCloseTo(28.125);
    expect(parseFloat(region.style.top)).toBeCloseTo(20.8333);
    expect(parseFloat(region.style.width)).toBeCloseTo(43.75);
    expect(parseFloat(region.style.height)).toBeCloseTo(58.3333);
    const heatmap = within(region).getByRole("presentation", { hidden: true });
    expect(heatmap).toHaveAttribute("src", expect.stringMatching(/^data:image\/png;base64,iVBOR/));
    expect(heatmap.style.opacity).toBe("0.9");
  });

  it("lets the user hide the heatmap and change its opacity", async () => {
    const user = setup();
    await selectPhoto(user);
    await respond(await analyse(user), 200, fixtures.prediction.ai());
    const slider = screen.getByRole("slider", { name: "Opacity" });
    fireEvent.change(slider, { target: { value: "30" } });
    expect((screen.getByTestId("heatmap-region").firstChild as HTMLElement).style.opacity).toBe("0.3");
    expect(slider).toHaveAttribute("aria-valuetext", "30%");
    await user.click(screen.getByRole("checkbox", { name: "Show heatmap" }));
    expect(screen.getByTestId("heatmap-region")).toBeEmptyDOMElement();
    expect(slider).toBeDisabled();
  });

  it("shows a real verdict without a heatmap when explanations are off", async () => {
    const user = setup();
    await selectPhoto(user);
    await user.click(screen.getByRole("checkbox", { name: /include heatmap/i }));
    const xhr = await analyse(user);
    expect(xhr.body?.get("explain")).toBe("false");
    await respond(xhr, 200, fixtures.prediction.real());
    expect(screen.getByRole("heading", { name: "Likely real" })).toHaveFocus();
    expect(screen.getByText("Not requested")).toBeInTheDocument();
    expect(screen.queryByTestId("heatmap-region")).not.toBeInTheDocument();
    expect(screen.queryByRole("group", { name: "Heatmap overlay" })).not.toBeInTheDocument();
  });

  it("explains an inconclusive result", async () => {
    const user = setup();
    await selectPhoto(user);
    await respond(await analyse(user), 200, fixtures.prediction.inconclusive());
    expect(screen.getByRole("heading", { name: "Inconclusive" })).toBeInTheDocument();
    expect(screen.getByText(/falls inside the uncertainty band of 20\.0% to 80\.0%/)).toBeInTheDocument();
    expect(screen.getByText(/treat this image as undetermined/i)).toBeInTheDocument();
  });

  it("shows the untrained notice when the prediction comes from the stand-in model", async () => {
    const user = setup();
    await selectPhoto(user);
    await respond(await analyse(user), 200, fixtures.prediction.untrained());
    expect(screen.getByRole("heading", { name: "The deployed model is an untrained stand-in" })).toBeInTheDocument();
    expect(screen.getByText(/single decision threshold at p\(ai\) = 50\.0%/i)).toBeInTheDocument();
  });

  it.each([
    ["missing_file", 400, "No image was received", true],
    ["empty_file", 400, "The file is empty", false],
    ["invalid_image", 400, "This file could not be read as an image", false],
    ["unsupported_format", 415, "Unsupported file type", false],
    ["animated_image", 415, "Animated images are not supported", false],
    ["invalid_request", 400, "The server could not read the upload", true],
    ["image_too_large", 413, "Image dimensions are too large", false],
    ["file_too_large", 413, "File is larger than 10 MB", false],
    ["rate_limited", 429, "Too many requests", true],
    ["model_unavailable", 503, "The detection model is unavailable", true],
    ["not_found", 404, "Analysis service not found", false],
    ["method_not_allowed", 405, "The service rejected this request", false],
    ["internal_error", 500, "The server hit an unexpected error", true],
  ])("maps API error %s to a friendly, focused message", async (code, status, title, retryable) => {
    const user = setup();
    await selectPhoto(user);
    await respond(await analyse(user), status, { error: { code, message: "raw" } });
    const alert = screen.getByRole("alert");
    expect(within(alert).getByRole("heading", { name: title })).toHaveFocus();
    expect(within(alert).queryByRole("button", { name: "Try again" }) !== null).toBe(retryable);
    expect(within(alert).getByRole("button", { name: "Choose another image" })).toBeInTheDocument();
  });

  it("recovers from a network failure with Try again", async () => {
    const user = setup();
    await selectPhoto(user);
    const first = await analyse(user);
    await fire(() => { first.onerror?.(); });
    expect(screen.getByRole("heading", { name: "Could not reach the server" })).toHaveFocus();
    await user.click(screen.getByRole("button", { name: "Try again" }));
    expect(FakeXHR.instances).toHaveLength(2);
    await respond(FakeXHR.latest(), 200, fixtures.prediction.ai());
    expect(screen.getByRole("heading", { name: "Likely AI-generated" })).toBeInTheDocument();
  });

  it("reports a timeout", async () => {
    const user = setup();
    await selectPhoto(user);
    const xhr = await analyse(user);
    await fire(() => { xhr.ontimeout?.(); });
    expect(screen.getByRole("heading", { name: "The request took too long" })).toBeInTheDocument();
  });

  it("cancels an analysis in flight", async () => {
    const user = setup();
    await selectPhoto(user);
    const xhr = await analyse(user);
    await user.click(screen.getByRole("button", { name: "Cancel" }));
    expect(xhr.aborted).toBe(true);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Analyse image" })).toBeEnabled();
  });

  it("resets to the drop zone and releases the preview", async () => {
    const user = setup();
    await selectPhoto(user);
    await respond(await analyse(user), 200, fixtures.prediction.ai());
    const revoke = vi.spyOn(URL, "revokeObjectURL");
    await user.click(screen.getByRole("button", { name: "Analyse another image" }));
    expect(screen.getByLabelText(/drop an image here/i)).toBeInTheDocument();
    expect(revoke).toHaveBeenCalledWith("blob:preview-1");
  });

  it("removes the selected image", async () => {
    const user = setup();
    await selectPhoto(user);
    await user.click(screen.getByRole("button", { name: "Remove image" }));
    expect(screen.queryByAltText("Preview of photo.jpg")).not.toBeInTheDocument();
  });

  it("rejects images over 40 megapixels once the browser has measured them", async () => {
    const user = setup();
    await selectPhoto(user, photo(), [9000, 5000]);
    expect(screen.getByRole("heading", { name: "Image dimensions are too large" })).toHaveFocus();
    expect(screen.getByText(/9000 × 5000 \(45\.0 megapixels\)/)).toBeInTheDocument();
  });

  it("rejects files the browser cannot decode", async () => {
    const user = setup();
    await user.upload(screen.getByLabelText(/drop an image here/i), photo());
    fireEvent.error(await screen.findByAltText("Preview of photo.jpg"));
    expect(screen.getByText(/the browser could not decode this file/i)).toBeInTheDocument();
  });

  it("ignores a stale validation when a newer file was chosen", async () => {
    setup();
    const zone = screen.getByLabelText(/drop an image here/i).closest(".dropzone") as Element;
    fireEvent.drop(zone, { dataTransfer: { files: [makeFile(asciiBytes("bad"), "bad.jpg")] } });
    fireEvent.drop(zone, { dataTransfer: { files: [makeFile(JPEG_BYTES, "good.jpg")] } });
    expect(await screen.findByAltText("Preview of good.jpg")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
