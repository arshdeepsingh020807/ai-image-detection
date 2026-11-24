import { act, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { JPEG_BYTES, makeFile } from "../test/files";
import { DropZone } from "./DropZone";

const zoneOf = () => screen.getByLabelText(/drop an image here/i).closest(".dropzone") as HTMLElement;

describe("DropZone", () => {
  it("is a labelled, keyboard-focusable file input restricted to supported types", async () => {
    render(<DropZone onFile={vi.fn()} busy={false} />);
    const input = screen.getByLabelText(/drop an image here/i);
    expect(input).toHaveAttribute("accept", expect.stringContaining("image/webp"));
    expect(input).toHaveAccessibleDescription("JPEG, PNG or WebP, up to 10 MB");
    await userEvent.tab();
    expect(input).toHaveFocus();
  });

  it("passes a picked file and clears the input so the same file can be picked again", async () => {
    const onFile = vi.fn();
    render(<DropZone onFile={onFile} busy={false} />);
    const input = screen.getByLabelText<HTMLInputElement>(/drop an image here/i);
    const file = makeFile(JPEG_BYTES);
    await userEvent.upload(input, file);
    expect(onFile).toHaveBeenCalledWith(file);
    expect(input.value).toBe("");
  });

  it("highlights while dragging and accepts the first dropped file", () => {
    const onFile = vi.fn();
    render(<DropZone onFile={onFile} busy={false} />);
    const zone = zoneOf();
    fireEvent.dragOver(zone, { dataTransfer: { dropEffect: "none" } });
    expect(zone).toHaveAttribute("data-dragging");
    const leave = (relatedTarget: EventTarget | null) => { zone.dispatchEvent(new MouseEvent("dragleave", { bubbles: true, relatedTarget })); };
    act(() => { leave(zone.querySelector("label")); });
    expect(zone).toHaveAttribute("data-dragging");
    act(() => { leave(document.body); });
    expect(zone).not.toHaveAttribute("data-dragging");
    const first = makeFile(JPEG_BYTES, "a.jpg");
    fireEvent.drop(zone, { dataTransfer: { files: [first, makeFile(JPEG_BYTES, "b.jpg")] } });
    expect(onFile).toHaveBeenCalledExactlyOnceWith(first);
  });

  it("ignores drops and empty picks without files", () => {
    const onFile = vi.fn();
    render(<DropZone onFile={onFile} busy={false} />);
    fireEvent.drop(zoneOf(), { dataTransfer: { files: [] } });
    fireEvent.change(screen.getByLabelText(/drop an image here/i), { target: { files: [] } });
    expect(onFile).not.toHaveBeenCalled();
  });

  it("disables input while a file is checked and links an error description", () => {
    render(<DropZone onFile={vi.fn()} busy describedBy="err" />);
    expect(screen.getByLabelText(/checking file/i)).toBeDisabled();
    expect(screen.getByLabelText(/checking file/i)).toHaveAttribute("aria-describedby", "dropzone-hint err");
  });
});
