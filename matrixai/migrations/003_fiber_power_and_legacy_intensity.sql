-- Light power through the fiber before and after coating (µW).
ALTER TABLE experiments
    ADD COLUMN IF NOT EXISTS fiber_power_before_uw FLOAT,
    ADD COLUMN IF NOT EXISTS fiber_power_after_uw  FLOAT;

-- Older runs kept their intensity in the legacy wavelength_shift column,
-- which the app never reads. Those values are µW (confirmed 2026-11-28),
-- so copy them across where nothing has been recorded yet. The legacy
-- column is left in place untouched.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns
                WHERE table_schema = current_schema() AND table_name = 'experiments'
                  AND column_name = 'wavelength_shift')
    THEN
        UPDATE experiments
           SET light_intensity_uw = wavelength_shift
         WHERE light_intensity_uw IS NULL AND wavelength_shift IS NOT NULL;
    END IF;
END $$;
