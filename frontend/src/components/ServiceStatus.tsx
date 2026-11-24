import { getHealth } from "../api/client";
import { useAsync } from "../hooks/useAsync";

const loadHealth = (signal: AbortSignal) => getHealth({ signal });

export function ServiceStatus() {
  const health = useAsync(loadHealth);
  const [tone, text] =
    health.status === "loading" ? ["pending", "Checking service…"]
      : health.status === "error" ? ["down", "Service unreachable"]
        : health.data.status === "ok"
          ? ["up", `Service online · model ${health.data.model_version ?? "loaded"}`]
          : ["down", "Service degraded · model not loaded"];
  return (
    <p className="service-status" data-tone={tone}>
      <span className="service-status__dot" aria-hidden="true" />
      {text}
    </p>
  );
}
