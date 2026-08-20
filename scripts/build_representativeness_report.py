"""Build the representativeness report (эпик 001, этап 5).

Compares every Слой 1 ЦОФ distribution over the full res-10 grid
(population) against the training sample (valuation objects joined to
their cells) — PSI + KS per feature, segments: overall + per asset
class. Writes ``data.parquet`` and a human-readable ``report.md`` under
``data/gold/representativeness/region={REGION}/resolution={R}/``.
"""

from kadastra.composition_root import Container
from kadastra.config import Settings
from kadastra.domain.asset_class import AssetClass


def main() -> None:
    settings = Settings()
    container = Container(settings)
    usecase = container.build_representativeness_report()

    classes = list(AssetClass)
    print(
        f"Representativeness report: region={settings.region_code} "
        f"resolution={settings.cell_tsorf_resolution} "
        f"classes={[c.value for c in classes]}"
    )
    report = usecase.execute(settings.region_code, classes)
    print(report.sort("psi", descending=True, nulls_last=True).head(10))
    print("done")


if __name__ == "__main__":
    main()
