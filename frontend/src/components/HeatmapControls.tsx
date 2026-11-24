import { heatmapGradient } from "../lib/heatmap";

interface HeatmapControlsProps {
  note: string;
  visible: boolean;
  opacity: number;
  onVisibleChange: (visible: boolean) => void;
  onOpacityChange: (opacity: number) => void;
}

export function HeatmapControls({ note, visible, opacity, onVisibleChange, onOpacityChange }: HeatmapControlsProps) {
  return (
    <fieldset className="heatmap-controls">
      <legend>Heatmap overlay</legend>
      <label className="checkbox">
        <input type="checkbox" checked={visible} onChange={(event) => { onVisibleChange(event.target.checked); }} />
        Show heatmap
      </label>
      <label className="range">
        <span>Opacity</span>
        <input
          type="range"
          min={0}
          max={100}
          step={5}
          value={Math.round(opacity * 100)}
          disabled={!visible}
          aria-valuetext={`${Math.round(opacity * 100)}%`}
          onChange={(event) => { onOpacityChange(Number(event.target.value) / 100); }}
        />
        <span className="range__value" aria-hidden="true">{Math.round(opacity * 100)}%</span>
      </label>
      <div className="heatmap-legend">
        <span className="heatmap-legend__bar" aria-hidden="true">
          <span className="heatmap-legend__ramp" style={{ backgroundImage: heatmapGradient() }} />
        </span>
        <span className="heatmap-legend__labels" aria-hidden="true"><span>Less</span><span>More influence</span></span>
      </div>
      <p className="heatmap-controls__note">{note} The outline marks the region the model analysed.</p>
    </fieldset>
  );
}
