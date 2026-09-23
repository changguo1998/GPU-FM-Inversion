if (ARGC != 4) {
    print "usage: gnuplot -c waveform_comparison.gp <output> <data-dir> <station-count> <station-ids>"
    exit error
}

station_count = int(ARG3)
station_ids = ARG4
image_height = 100 + 140 * station_count

stats sprintf("%s/001_P.dat", ARG2) using 1 nooutput
p_min = STATS_min
p_max = STATS_max
stats sprintf("%s/001_S.dat", ARG2) using 1 nooutput
s_min = STATS_min
s_max = STATS_max
p_duration = p_max - p_min
s_duration = s_max - s_min

left_margin = 0.06
right_margin = 0.995
bottom_margin = 0.015
top_margin = 0.985
horizontal_gap = 0.006
vertical_gap = 0.004
usable_width = right_margin - left_margin - 5 * horizontal_gap
duration_sum = 3 * p_duration + 3 * s_duration
p_width = usable_width * p_duration / duration_sum
s_width = usable_width * s_duration / duration_sum
panel_height = (top_margin - bottom_margin - (station_count - 1) * vertical_gap) / station_count

set terminal pngcairo size 1800,image_height enhanced font "DejaVu Sans,10"
set output ARG1
set multiplot

set yrange [-0.85:0.85]
unset border
unset xtics
unset ytics
unset grid
unset xlabel
unset ylabel

do for [station = 1:station_count] {
    panel_origin_x = left_margin
    panel_origin_y = top_margin - station * panel_height - (station - 1) * vertical_gap
    do for [panel = 1:6] {
        phase_index = int((panel - 1) / 3) + 1
        component_index = (panel - 1) % 3 + 1
        phase = phase_index == 1 ? "P" : "S"
        panel_width = phase_index == 1 ? p_width : s_width
        component = word("Z N E", component_index)
        observed_column = 2 * component_index
        synthetic_column = observed_column + 1
        data_file = sprintf("%s/%03d_%s.dat", ARG2, station, phase)
        set origin panel_origin_x,panel_origin_y
        set size panel_width,panel_height
        if (phase_index == 1) {
            set xrange [p_min:p_max]
        } else {
            set xrange [s_min:s_max]
        }

        if (station == 1) {
            set title sprintf("%s-%s", phase, component) font "DejaVu Sans,12"
        } else {
            unset title
        }
        if (panel == 1) {
            set label 1 word(station_ids, station) at graph -0.04,0.5 right \
                font "DejaVu Sans,11"
        } else {
            unset label 1
        }
        if (station == 1 && panel == 1) {
            set key horizontal top right samplen 2 font "DejaVu Sans,10"
        } else {
            unset key
        }

        plot data_file using 1:(column(observed_column)) with lines linewidth 1.8 linecolor rgb "#222222" title "Observed", \
             data_file using 1:(column(synthetic_column)) with lines linewidth 1.8 linecolor rgb "#d73027" title "Synthetic"

        panel_origin_x = panel_origin_x + panel_width + horizontal_gap
    }
}

unset multiplot
