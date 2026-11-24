export interface Stat {
  label: string;
  value: string;
  hint?: string | undefined;
}

export function StatTiles({ stats, label }: { stats: Stat[]; label: string }) {
  return (
    <dl className="stat-tiles" aria-label={label}>
      {stats.map((stat) => (
        <div key={stat.label} className="stat-tile">
          <dt>{stat.label}</dt>
          <dd>
            <span className="stat-tile__value num">{stat.value}</span>
            {stat.hint && <span className="stat-tile__hint">{stat.hint}</span>}
          </dd>
        </div>
      ))}
    </dl>
  );
}
