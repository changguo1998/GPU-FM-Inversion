if (ARGC != 3) {
    print "usage: gnuplot -c event_stations.gp <output> <stations.dat> <event.dat>"
    exit error
}

stats ARG2 using 1 nooutput
station_min_x = STATS_min
station_max_x = STATS_max
stats ARG2 using 2 nooutput
station_min_y = STATS_min
station_max_y = STATS_max
stats ARG3 using 1 nooutput
event_x = STATS_min
stats ARG3 using 2 nooutput
event_y = STATS_min

min_x = station_min_x < event_x ? station_min_x : event_x
max_x = station_max_x > event_x ? station_max_x : event_x
min_y = station_min_y < event_y ? station_min_y : event_y
max_y = station_max_y > event_y ? station_max_y : event_y
x_extent = max_x - min_x
y_extent = max_y - min_y
outer_radius = 0.035 * (x_extent > y_extent ? x_extent : y_extent)
inner_radius = outer_radius * 0.381966

set terminal pngcairo size 1200,900 enhanced font "DejaVu Sans,12"
set output ARG1
set title "Event and station distribution"
set xlabel "Longitude (deg)"
set ylabel "Latitude (deg)"
set grid
set size ratio -1
set key outside

set object 1 polygon \
    from event_x + outer_radius * cos(pi / 2), event_y + outer_radius * sin(pi / 2) \
    to event_x + inner_radius * cos(7 * pi / 10), event_y + inner_radius * sin(7 * pi / 10) \
    to event_x + outer_radius * cos(9 * pi / 10), event_y + outer_radius * sin(9 * pi / 10) \
    to event_x + inner_radius * cos(11 * pi / 10), event_y + inner_radius * sin(11 * pi / 10) \
    to event_x + outer_radius * cos(13 * pi / 10), event_y + outer_radius * sin(13 * pi / 10) \
    to event_x + inner_radius * cos(3 * pi / 2), event_y + inner_radius * sin(3 * pi / 2) \
    to event_x + outer_radius * cos(17 * pi / 10), event_y + outer_radius * sin(17 * pi / 10) \
    to event_x + inner_radius * cos(19 * pi / 10), event_y + inner_radius * sin(19 * pi / 10) \
    to event_x + outer_radius * cos(pi / 10), event_y + outer_radius * sin(pi / 10) \
    to event_x + inner_radius * cos(3 * pi / 10), event_y + inner_radius * sin(3 * pi / 10) \
    to event_x + outer_radius * cos(pi / 2), event_y + outer_radius * sin(pi / 2) \
    fillcolor rgb "#b2182b" fillstyle solid 1.0 border linecolor rgb "#b2182b" front

plot ARG2 using 1:2 with points pointtype 9 pointsize 2.0 linecolor rgb "#2166ac" title "Stations", \
     "" using 1:2:3 with labels offset 0.7,0.5 notitle, \
     ARG3 using 1:2:(sprintf("Event M%.1f", column(3))) with labels offset 2.0,1.5 notitle
