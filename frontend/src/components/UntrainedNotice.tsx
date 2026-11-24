import { Notice } from "./Notice";

export function UntrainedNotice({ version }: { version: string }) {
  return (
    <Notice tone="warning" level={2} title="The deployed model is an untrained stand-in">
      <p>
        Model <code>{version}</code> has not been trained. It exists so the service can be tested end to end, and its outputs
        carry no information about whether an image is real or AI-generated. Do not interpret any result until a trained model
        is deployed.
      </p>
    </Notice>
  );
}
