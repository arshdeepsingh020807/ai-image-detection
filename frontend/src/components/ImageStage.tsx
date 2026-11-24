import { useState, type SyntheticEvent } from "react";
import type { Explanation } from "../api/types";
import { containBox, overlayBox, toPercentStyle } from "../lib/region";

export const STAGE_ASPECT = 4 / 3;

interface ImageStageProps {
  src: string;
  alt: string;
  explanation: Explanation | null;
  showHeatmap: boolean;
  opacity: number;
  onDimensions: (width: number, height: number) => void;
  onDecodeError: () => void;
}

export function ImageStage({ src, alt, explanation, showHeatmap, opacity, onDimensions, onDecodeError }: ImageStageProps) {
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  const handleLoad = (event: SyntheticEvent<HTMLImageElement>) => {
    const { naturalWidth: width, naturalHeight: height } = event.currentTarget;
    setSize({ width, height });
    onDimensions(width, height);
  };
  const frame = size ? containBox(STAGE_ASPECT, size.width / size.height) : null;
  const overlay = frame && explanation ? overlayBox(frame, explanation.region) : null;

  return (
    <div className="stage" data-loaded={frame ? "" : undefined}>
      <img
        className="stage__image"
        src={src}
        alt={alt}
        style={frame ? toPercentStyle(frame) : undefined}
        onLoad={handleLoad}
        onError={onDecodeError}
      />
      {overlay && explanation && (
        <div className="stage__overlay" style={toPercentStyle(overlay)} data-testid="heatmap-region">
          {showHeatmap && (
            <img
              className="stage__heatmap"
              src={`data:image/png;base64,${explanation.heatmap_png_base64}`}
              alt=""
              style={{ opacity }}
            />
          )}
        </div>
      )}
    </div>
  );
}
