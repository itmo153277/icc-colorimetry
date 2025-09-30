gamescope.config.known_displays.hp_x27q = {
    pretty_name = "HP X27q",
    colorimetry = {
        r = { x = 0.706, y = 0.329 },
        g = { x = 0.301, y = 0.627 },
        b = { x = 0.133, y = 0.033 },
        w = { x = 0.317, y = 0.333 },
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
