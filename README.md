# Global Energy Stress OS

完全無料データソースを使ったエネルギー・海運・地政学ストレス監視OSです。

## Phase 1

- Yahoo Financeから市場データ取得
- Google News RSS / GDELT / 公開RSSからニュース件数取得
- War Premium Score算出
- Hormuz Closure Probability算出
- CSV / Markdown / HTML / PNGを自動生成
- GitHub Actionsで毎朝5:00 JSTに自動実行
- GitHub Pages用に `docs/` を出力

## GitHub Actions

`.github/workflows/daily.yml` は JST 05:00 に実行されます。

手動実行する場合:

Actions → Global Energy Stress OS → Run workflow

## 出力

- `outputs/dashboard.csv`
- `outputs/news_counts.csv`
- `outputs/dashboard.md`
- `outputs/dashboard.html`
- `outputs/dashboard.png`
- `docs/index.html`

## 注意

このPhase 1はルールベースの無料データ試作版です。OllamaによるLLM分類、Shipping Layer、Physical Layerは次フェーズで追加します。
