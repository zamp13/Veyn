set style data histogram
set style histogram cluster gap 2

set style fill solid border rgb "black"
set key inside
set yrange [20:95]
set xtics rotate by 45 right
set datafile separator "\t"


set term png
set output "../results/tsv/afterVoteMajoritaire/CRFvsRNNneverSeen.png"


plot '../results/tsv/afterVoteMajoritaire/CRFvsRNNneverSeen.tsv' using 2:xtic(1) title col(2) linecolor rgb "#87CEFA",\
        '' using 3 title col(3) linecolor rgb "#00BFFF",\
        '' using 4 title col(4) linecolor rgb "cyan",\
	'' using 5 title col(5) linecolor rgb "white",\
	'' using 6 title col(6) linecolor rgb "#0000CD",\
        '' using 7 title col(7) linecolor rgb "#00008B",\
	'' using 8 title col(8) linecolor rgb "blue"

