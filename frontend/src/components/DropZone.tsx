import { useState, type ChangeEvent, type DragEvent } from "react";
import { ACCEPT } from "../lib/fileValidation";

interface DropZoneProps {
  onFile: (file: File) => void;
  busy: boolean;
  describedBy?: string | undefined;
}

export function DropZone({ onFile, busy, describedBy }: DropZoneProps) {
  const [dragging, setDragging] = useState(false);

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) onFile(file);
  };
  const handleDragOver = (event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setDragging(true);
  };
  const handleDragLeave = (event: DragEvent) => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setDragging(false);
  };
  const handleDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) onFile(file);
  };

  return (
    <div
      className="dropzone"
      data-dragging={dragging || undefined}
      data-busy={busy || undefined}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <input
        id="image-input"
        className="visually-hidden"
        type="file"
        accept={ACCEPT}
        onChange={handleChange}
        disabled={busy}
        aria-describedby={["dropzone-hint", describedBy].filter(Boolean).join(" ")}
      />
      <label htmlFor="image-input" className="dropzone__label">
        <span className="dropzone__title">{busy ? "Checking file…" : "Drop an image here"}</span>
        <span className="dropzone__action">or <span className="dropzone__link">choose a file</span></span>
        <span className="dropzone__hint" id="dropzone-hint">JPEG, PNG or WebP, up to 10 MB</span>
      </label>
    </div>
  );
}
