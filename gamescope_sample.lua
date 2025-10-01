gamescope.config.known_displays.hp_x27q = {
    pretty_name = "HP X27q",
    colorimetry = {
        r = { x = 0.6855, y = 0.3085 },
        g = { x = 0.2646, y = 0.6679 },
        b = { x = 0.1503, y = 0.0576 },
        w = { x = 0.3177, y = 0.3343 },
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
