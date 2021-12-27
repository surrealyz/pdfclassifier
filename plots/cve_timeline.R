library(ggplot2)
library(scales)
library(grid)

fname <- "cve_timeline.csv"
df <- read.table(fname, col.names=c("year", "count"), sep=',')

p <- ggplot(df) +
    geom_line(aes(x=year, y=count), size=1) +
    geom_point(aes(x=year, y=count), shape=1, size=3) +
    ylim(0, 200) +
    xlab(NULL) +
    ylab("Number of CVEs") +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),panel.background = element_blank(),panel.border = element_rect(fill = NA, colour = "black"),
          text=element_text(size=26), axis.text.x=element_text(size=26),axis.text.y=element_text(size=26),
          legend.position=c(0.7, 0.6), legend.title=element_blank(),
          legend.box.background = element_rect(fill = NA, linetype="solid", size=0.5),
          legend.key.width = unit(1.2, "cm"))

ggsave("cve_timeline.pdf", plot = p, width=10, height=4)
ggsave("cve_timeline.png", plot = p, width=10, height=4)
