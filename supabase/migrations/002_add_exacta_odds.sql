-- predictions テーブルに exacta_odds 列を追加
-- ◎→この選手 の2連単オッズ（rank 2-5の選手に設定）
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS exacta_odds DOUBLE PRECISION;

-- prediction_detail ビューを更新（exacta_odds を含める）
CREATE OR REPLACE VIEW prediction_detail
WITH (security_invoker = on) AS
SELECT
    p.race_id,
    r.date AS race_date,
    r.velodrome,
    r.velodrome_code,
    r.race_number,
    r.race_name,
    r.grade,
    r.bank_length,
    r.rider_count,
    r.start_time,
    p.rider_id,
    COALESCE(e.rider_name, ri.name) AS rider_name,
    COALESCE(e.class, ri.class) AS rider_class,
    ri.prefecture,
    e.frame_number,
    e.bike_number,
    e.avg_competition_score,
    e.win_rate,
    e.place_rate,
    e.gear_ratio,
    p.predicted_score,
    p.predicted_rank,
    p.mark,
    p.confidence,
    p.exacta_odds,
    rr.finish_position,
    rr.last_1lap,
    rr.odds AS result_odds,
    rr.backstretching
FROM predictions p
JOIN races r ON r.race_id = p.race_id
LEFT JOIN riders ri ON ri.rider_id = p.rider_id
LEFT JOIN entries e ON e.race_id = p.race_id AND e.rider_id = p.rider_id
LEFT JOIN race_results rr ON rr.race_id = p.race_id AND rr.rider_id = p.rider_id
ORDER BY r.date DESC, r.race_number, p.predicted_rank;
