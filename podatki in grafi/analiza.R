## 1. igra vrednosti
iteracija1_1_vrednosti <- read.csv("~/magisterij/matematika z računalnikom/Iterative-computation-of-Nash-equilibrium/iteracija1_matrika0_100_vrednosti.txt", header=FALSE)
iteracija1_1_vrednosti <- t(iteracija1_1_vrednosti)
iteracija1_1_vrednosti<- ts(iteracija1_1_vrednosti, start = 1)

iteracija2_1_vrednosti <- read.csv("~/magisterij/matematika z računalnikom/Iterative-computation-of-Nash-equilibrium/iteracija_brown_matrika0_100.txt", header=FALSE)
iteracija2_1_vrednosti <- t(iteracija2_1_vrednosti)
iteracija2_1_vrednosti<- ts(iteracija2_1_vrednosti, start = 1)
plot(iteracija1_1_vrednosti, xlim = c(1, 100), ylim=c(1.9,3.8),col= "red", type = "l", lwd=0.5, xlab = "Korak iteracije", ylab="Vrednost 1.igre")
legend("topright",
       c("1. algoritem","2. algoritem"),
       col=c("red","blue"),
       lty="solid",
       bty="n",
       lwd =0.5)

lines(iteracija2_1_vrednosti, col = "blue", type="l", lwd=0.5)
abline(h=(10/3))


##2. igra vrednosti
iteracija1_2_vrednosti <- read.csv("~/magisterij/matematika z računalnikom/Iterative-computation-of-Nash-equilibrium/iteracija1_matrika1_1000_vrednosti.txt", header=FALSE)
iteracija1_2_vrednosti <- t(iteracija1_2_vrednosti)
iteracija1_2_vrednosti<- ts(iteracija1_2_vrednosti, start = 1)

iteracija2_2_vrednosti <- read.csv("~/magisterij/matematika z računalnikom/Iterative-computation-of-Nash-equilibrium/iteracija_brown_matrika1_15000_sp.txt", header=FALSE)
iteracija2_2_vrednosti <- t(iteracija2_2_vrednosti)
iteracija2_2_vrednosti<- ts(iteracija2_2_vrednosti, start = 1)
plot(iteracija1_2_vrednosti[1:500, 1], xlim = c(1, 100), ylim=c(-1, 2),col= "red", type = "l", lwd=0.5, xlab = "Korak iteracije", ylab="Vrednost 2.igre")
legend("topright",
       c("1. algoritem","2. algoritem"),
       col=c("red","blue"),
       lty="solid",
       bty="n",
       lwd =0.5)

lines(iteracija2_2_vrednosti[1:500, 1], col = "blue", type="l", lwd=0.5)
abline(h=0)

##1. igra - napaka v odvisnosti od korakov

iteracija1_1_napaka <- read.csv("~/magisterij/matematika z računalnikom/Iterative-computation-of-Nash-equilibrium/iteracija1_matrika0_napake.txt", header=FALSE)
iteracija1_1_napaka <- t(iteracija1_1_napaka)
iteracija1_1_napaka<- ts(iteracija1_1_napaka, start = 1)

iteracija2_1_napaka <- read.csv("~/magisterij/matematika z računalnikom/Iterative-computation-of-Nash-equilibrium/iteracija_brown_matrika0_napaka.txt", header=FALSE)
iteracija2_1_napaka <- t(iteracija2_1_napaka)
iteracija2_1_napaka<- ts(iteracija2_1_napaka, start = 1)

plot(iteracija1_1_napaka, col= "red", ylim= c(-0.1, 1.9),type = "l", lwd=0.5, xlab = "Korak iteracije", ylab="Napaka 1.igre", main="Napaka v odvisnosti od korakov iteracije")
legend("topright",
       c("1.algoritem","2.algoritem"),
       col=c("red","blue"),
       lty="solid",
       bty="n",
       lwd =0.5)

lines(iteracija2_1_napaka, col = "blue", type="l", lwd=0.5)
##2. igra - napaka v odvisnosti od korakov


iteracija1_2_napaka <- read.csv("~/magisterij/matematika z računalnikom/Iterative-computation-of-Nash-equilibrium/iteracija1_matrika1_napake.txt", header=FALSE)
iteracija1_2_napaka <- t(iteracija1_2_napaka)
iteracija1_2_napaka<- ts(iteracija1_2_napaka, start = 1)

iteracija2_2_napaka <- read.csv("~/magisterij/matematika z računalnikom/Iterative-computation-of-Nash-equilibrium/iteracija_brown_matrika2_napaka.txt", header=FALSE)
iteracija2_2_napaka <- t(iteracija2_2_napaka)
iteracija2_2_napaka<- ts(iteracija2_2_napaka, start = 1)

plot(iteracija1_2_napaka[1:100,1], col= "red", type = "l", lwd=0.5, xlab = "Korak iteracije", ylab="Napaka 2.igre", main="Napaka v odvisnosti od korakov iteracije")
legend("topright",
       c("1.algoritem","2.algoritem"),
       col=c("red","blue"),
       lty="solid",
       bty="n",
       lwd =0.5)

lines(iteracija2_2_napaka, col = "blue", type="l", lwd=0.5)
