gamescope.config.known_displays.hp_x27q = {
    pretty_name = "HP X27q",
    colorimetry = {
        r = { x = 0.7051, y = 0.3311 },
        g = { x = 0.2990, y = 0.6280 },
        b = { x = 0.1316, y = 0.0327 },
        w = { x = 0.3174, y = 0.3330 },
    },
    matches = function(display)
        local lcd_types = {
            { vendor = "HPN", model = "HP X27q" },
        }

        for index, value in ipairs(lcd_types) do
            if value.vendor == display.vendor and value.model == display.model then
                debug("[custom] Matched vendor: "..value.vendor.." model: "..value.model)
                return 5000
            end
        end
        return -1
    end
}
