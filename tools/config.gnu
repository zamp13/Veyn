set style data histogram
set style histogram cluster gap 1 errorbars linewidth 1.5

set style fill solid border rgb "black"
set auto x
set grid y
set key outside top right
set yrange [0:100]
set xtics rotate
set datafile separator "\t"


set term png
set output "../results/tsv/diffLanguage.png"


plot '../results/tsv/diffLanguage.tsv' using 2:3:xtic(1) title col(2) linecolor rgb "grey30",\
        '' using 4:5 title col(4) linecolor rgb "white",\
        '' using 6:7 title col(6) linecolor rgb "grey60"
