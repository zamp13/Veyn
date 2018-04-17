set style data histogram
set style histogram cluster gap 1

set style fill solid border rgb "black"
set auto x
set grid y
set key inside
set yrange [20:70]
set xtics rotate by 45 right
set datafile separator "\t"


set term png
set output "../results/tsv/afterVoteMajoritaire/embedSlides.png"

plot '../results/tsv/afterVoteMajoritaire/embedSlides.tsv' using 2:xtic(1) title col(2)  linecolor rgb "cyan",\
        '' using 3 title col(3) linecolor rgb "blue"\

