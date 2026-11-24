export function LimitationsNote() {
  return (
    <aside className="limitations" aria-labelledby="limitations-title">
      <h2 id="limitations-title">How to read a result</h2>
      <ul>
        <li><strong>A statistical estimate, not proof.</strong> The verdict is a calibrated probability from a classifier and it can be wrong in either direction.</li>
        <li><strong>The heatmap shows model attention, not evidence.</strong> It highlights the regions the classifier relied on; it does not mark edited or generated pixels.</li>
        <li><strong>Training data limits what it knows.</strong> The model learned from face photographs and a general-content set. Other content, new generators, heavy compression, resizing or screenshots can reduce accuracy.</li>
        <li><strong>Only the centre of the image is analysed.</strong> The image is scaled and centre-cropped to a square; the outlined region shows exactly what the model saw.</li>
        <li><strong>Do not use it as the sole basis for decisions about people.</strong></li>
      </ul>
    </aside>
  );
}
