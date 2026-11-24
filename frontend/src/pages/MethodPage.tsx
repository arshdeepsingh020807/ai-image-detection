import { Link } from "react-router-dom";
import { PageHeader } from "../components/PageHeader";
import { PipelineDiagram } from "../components/PipelineDiagram";

const SECTIONS = [
  ["data", "Data"],
  ["cleaning", "Cleaning and leakage prevention"],
  ["training", "Transfer learning"],
  ["calibration", "Calibration"],
  ["band", "The inconclusive band"],
  ["gradcam", "Grad-CAM heatmaps"],
  ["serving", "Serving"],
  ["limitations", "Limitations"],
] as const;

export function MethodPage() {
  return (
    <div className="page method-page">
      <PageHeader title="Method" eyebrow="How it works">
        <p>
          How the detector is built and why each step exists. Measured results live on the <Link to="/performance">Performance</Link> page.
        </p>
      </PageHeader>

      <PipelineDiagram />

      <div className="prose-layout">
        <nav className="report-contents" aria-label="Sections">
          <p className="report-contents__title">Sections</p>
          <ul>{SECTIONS.map(([id, label]) => <li key={id}><a href={`#${id}`}>{label}</a></li>)}</ul>
        </nav>
        <div className="prose">
          <section id="data" aria-labelledby="data-title">
            <h2 id="data-title">Data</h2>
            <p>
              The training pool combines the DeepDetect-2025 face dataset with a general-content dataset of real photographs and
              AI-generated images. Further sources are held out entirely: they are never used for training or model selection and serve
              only as external test sets, which show how well the detector transfers to images unlike its training data.
            </p>
          </section>

          <section id="cleaning" aria-labelledby="cleaning-title">
            <h2 id="cleaning-title">Cleaning and leakage prevention</h2>
            <p>Public image datasets contain duplicates, mislabelled files and the same picture saved twice. Left alone, these inflate test scores. Each image is audited and removed if any rule applies, in this order:</p>
            <ol>
              <li><strong>Unreadable or unsupported</strong> files, judged by content rather than file name, using the same decoder as the live service.</li>
              <li><strong>Exact duplicates</strong>, identified by an MD5 hash of the file bytes.</li>
              <li><strong>Conflicting labels</strong>: near-duplicate clusters that contain both real and AI-generated images.</li>
              <li><strong>External leaks</strong>: training images whose near-duplicate cluster touches an external test image.</li>
            </ol>
            <p>
              Near-duplicates are found with a 64-bit perceptual hash; images within a few bits of each other are clustered. Whole clusters
              are then assigned to the training, validation or test split, stratified by source and class, so a resized copy of a test image
              can never sit in the training set. A final check re-derives the clusters and counts any group that spans two splits.
            </p>
            <p>
              <strong>Shortcut probes</strong> train a simple classifier on metadata alone (file format, size, JPEG quality, average colour).
              If metadata predicts the label, a deep model might learn that shortcut instead of genuine generation traces, so the probe
              results are published alongside the dataset report.
            </p>
          </section>

          <section id="training" aria-labelledby="training-title">
            <h2 id="training-title">Transfer learning</h2>
            <p>
              The detector starts from a convolutional network pretrained on ImageNet. Its classifier is replaced by global average
              pooling, dropout and a single output: the logit for “AI-generated”. Training runs in two phases. First only the new head is
              trained while the backbone stays frozen; then the backbone is unfrozen and fine-tuned at a lower learning rate with a cosine
              schedule, stopping early when validation loss stops improving.
            </p>
            <p>
              Every image is resized so its short side is 256 px and centre-cropped to 224 × 224 px. Training adds random crops and
              augmentation applied identically to both classes, such as JPEG recompression, down- and re-sampling, horizontal flips and
              mild colour changes. Rotation, blur and strong colour shifts are deliberately left out because they erase the fine,
              high-frequency traces that generators leave. The same preprocessing code runs in training and in the live service.
            </p>
          </section>

          <section id="calibration" aria-labelledby="calibration-title">
            <h2 id="calibration-title">Calibration</h2>
            <p>
              A network's raw scores are usually overconfident. Temperature scaling divides the logit by a single constant <var>T</var> before
              the sigmoid, <code>p(AI) = sigmoid(logit / T)</code>, with <var>T</var> chosen on the validation split to minimise log loss. It does
              not change which images are ranked above others, but it makes the reported probability match how often the model is actually right at that level.
            </p>
          </section>

          <section id="band" aria-labelledby="band-title">
            <h2 id="band-title">The inconclusive band</h2>
            <p>
              Instead of forcing every image into a class, the detector abstains when the probability is close to the middle. Two cut-offs
              are chosen on validation data: the upper one is the lowest value that keeps the share of real images wrongly called
              AI-generated within a target error rate, and the lower one does the same for AI-generated images called real. Probabilities
              at or below the lower cut-off are reported as real, at or above the upper one as AI-generated, and anything between as
              inconclusive.
            </p>
          </section>

          <section id="gradcam" aria-labelledby="gradcam-title">
            <h2 id="gradcam-title">Grad-CAM heatmaps</h2>
            <p>
              Because the head is pooling followed by one linear layer, Grad-CAM reduces exactly to a weighted sum of the final feature maps:
              <code> ReLU(Σ w<sub>c</sub> · A<sub>c</sub>)</code>. The service computes it from the exported model's outputs, normalises it,
              upsamples it to 224 × 224 px and colours it with the upper half of the inferno colour map: transparent purple for little influence, rising to bright, more opaque pale yellow for the most. On the Detect page it is placed over the exact centre
              crop the model analysed.
            </p>
            <p>A heatmap shows where the model looked, not what is fake. Bright regions are not evidence of manipulation.</p>
          </section>

          <section id="serving" aria-labelledby="serving-title">
            <h2 id="serving-title">Serving</h2>
            <p>
              The trained network is exported to ONNX and run on CPU behind a small API. Uploads are checked by content, limited to 10 MB
              and 40 megapixels, and rejected if animated. This page checks the same rules before uploading so most mistakes are caught
              immediately.
            </p>
          </section>

          <section id="limitations" aria-labelledby="limitations-title-method">
            <h2 id="limitations-title-method">Limitations</h2>
            <ul>
              <li>Results are statistical estimates and can be wrong in either direction; they are not proof.</li>
              <li>New generators, unusual content, screenshots, heavy compression or editing can all reduce accuracy without warning.</li>
              <li>Only the centre crop is analysed, so content near the edges of a wide image is ignored.</li>
              <li>Benchmark scores describe the test data, not every image a user might upload; the external sets are the better guide.</li>
              <li>The tool should never be the sole basis for a decision about a person.</li>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );
}
