###########################################################
## Load and display data
# rm(list = ls())

library(ggplot2)
library(ggExtra)

dat = read.csv("TestPhase.csv")

## Expression bias
FixDur <- dat$fixDur
Emo <- dat$Emo2
Famil <- dat$Famil
Grp <- dat$Grp

# create a ggplot2 scatterplot
df <-
  data.frame(x = FixDur[Emo == "FEAR"], y = FixDur[Emo == "HAPPY"])
p <- ggplot(df,aes(x = x,y = y)) +
  stat_density2d(aes(fill = ..level..,alpha = ..level..),geom = 'polygon',colour =
                   'black') +
  scale_fill_continuous(low = "green",high = "red") +
  guides(alpha = "none")  + geom_point(colour = "black",alpha = 0.5) +
  xlab("Fear") + ylab("Happy") + xlim(-.1, 4.75) + ylim(-.1, 4.75)
# add marginal histograms
ggMarginal(p, type = "histogram",binwidth = 5 / 25)


df <-
  data.frame(x = FixDur[Famil == "famil"], y = FixDur[Famil == "novel"])
p <- ggplot(df,aes(x = x,y = y)) +
  stat_density2d(aes(fill = ..level..,alpha = ..level..),geom = 'polygon',colour =
                   'black') +
  scale_fill_continuous(low = "green",high = "red") +
  guides(alpha = "none")  + geom_point(colour = "black",alpha = 0.5) +
  xlab("Famil") + ylab("Novel") + xlim(-.1, 4.75) + ylim(-.1, 4.75)
# add marginal histograms
ggMarginal(p, type = "histogram",binwidth = 5 / 25)



###########################################################
## Generalized Additive mixed model with eye movement data
library(gamm4)
library(mgcv)
set.seed(0)
dat = read.csv("dstmp1.csv")

FixN <- dat$y
fear <- dat$i1
happy <- dat$j1

ba1 <- gamm4(FixN ~ s(fear,happy,k = 40),family = poisson)
plot(ba1$gam,pages = 1)

dat2 = read.csv("dstmp2.csv")

FixN <- dat2$y
famil <- dat2$i1
novel <- dat2$j1

ba2 <- gamm4(FixN ~ s(famil,novel,k = 40),family = poisson)
plot(ba2$gam,pages = 1)


##
br <- gamm(FixN ~ s(fear) + s(happy) + s(fear,happy),family = poisson)
plot(br$gam)


##
dat = read.csv("dstmp_c1.csv")

FixN <- dat$y
fear <- dat$i1
happy <- dat$j1
Grp <- factor(dat$Grp)
Emo <- factor(dat$Emo)
Fctype <- factor(dat$Fctype)

ba1 <- gamm4(FixN ~ s(fear,happy,k = 40),family = poisson)
plot(ba1$gam,pages = 1)

ba2 <-
  gamm4(FixN ~ s(fear,happy,k = 40) + Grp * Emo * Fctype,family = poisson)
plot(ba2$gam,pages = 1)
anova(ba2$gam)

ba3 <-
  gamm4(FixN ~ s(fear,happy,k = 40) + Grp + Emo + Fctype,family = poisson)
plot(ba3$gam,pages = 1)
anova(ba3$gam)

ba4 <- gamm4(FixN ~ s(fear,happy,k = 40) + Emo,family = poisson)
anova(ba4$gam)



###########################################################
## Bayesian (rjags)
rm(list = ls())

# Choose a condition:
condi <- 1
# Choose a model: 1 - estimate using two linked beta; 
#                 2 - estimate using Dirichlet
modelmod <- 1


library(R2jags)
library(ggplot2)
library(ggExtra)

dat = read.csv("TestPhase.csv")

## Expression bias
FixDur <- dat$fixDur
Emo <- dat$Emo2
Famil <- dat$Famil
Grp <- dat$Grp

# The datasets:
if (condi == 1) {
  x <- FixDur[Emo == "FEAR"]
  y <- FixDur[Emo == "HAPPY"]
  label1 <- "FEAR"
  label2 <- "HAPPY"
} else {
  x <- FixDur[Famil == "famil"]
  y <- FixDur[Famil == "novel"]
  label1 <- "Famil"
  label2 <- "Novel"
}

n <- length(x) # number of people/units measured
if (modelmod == 1) {
  dur <- (x + y) / 5
  x1 <- x / (x + y)
  
  x1[x1 == 1] = .999999999999999
  dur[dur == 1] = .999999999999999
  x1[x1 == 0] = 1 - .999999999999999
  dur[dur == 0] = 1 - .999999999999999
  
  data <- list(x = x1, n = n,dur = dur) # to be passed on to JAGS
  myinits <- list(list(
    phi1 = 2.9,a1 = 0,phi2 = 2.9,a2 = 0
  ))
  
  # parameters to be monitored:
  parameters <- c("alpha1", "beta1","mu1","alpha2", "beta2","mu2")
  
  # The following command calls JAGS with specific options.
  # For a detailed description see the R2jags documentation.
  samples <- jags(
    data, inits = myinits, parameters,
    model.file = "model1.txt", n.chains = 1, n.iter = 10000,
    n.burnin = 500, n.thin = 1, DIC = T
  )
  
  durtmp <- samples$BUGSoutput$sims.list$mu1
  durtrace <- durtmp * 5
  xtmp <- samples$BUGSoutput$sims.list$mu2
  xtrace <- durtrace * xtmp
  ytrace <- durtrace - xtrace
} else {
  z = 5 - x - y
  x[x <= 0] = 5 - 4.999999999999999
  y[y <= 0] = 5 - 4.999999999999999
  z[z <= 0] = 5 - 4.999999999999999
  
  data <- list(x = x, y = y, z = z, n = n) # to be passed on to JAGS
  myinits <- list(list(alpha1 = mean(x),alpha2 = mean(y)))
  
  # parameters to be monitored:
  parameters <- c("alpha1", "alpha2", "alpha3")
  
  # The following command calls JAGS with specific options.
  # For a detailed description see the R2jags documentation.
  samples <- jags(
    data, inits = myinits, parameters,
    model.file = "model2.txt", n.chains = 1, n.iter = 10000,
    n.burnin = 500, n.thin = 1, DIC = T
  )
  
  xtrace <- samples$BUGSoutput$sims.list$alpha1
  ytrace <- samples$BUGSoutput$sims.list$alpha2
}

library(lattice)
# Additional option: use some plots in coda
# first use as.mcmmc to convert rjags object into mcmc.list:
samples.mcmc <- as.mcmc(samples)
# then use the plotting methods from coda:
xyplot(samples.mcmc)
densityplot(samples.mcmc)

library(ggplot2)
library(ggExtra)
# create a ggplot2 scatterplot
df2 <- data.frame(x = xtrace, y = ytrace)
df <- data.frame(x = x, y = y)
p <- ggplot(df,aes(x = x,y = y)) +
  stat_density2d(aes(fill = ..level..,alpha = ..level..),geom = 'polygon',colour =
                   'black') +
  scale_fill_continuous(low = "green",high = "red") +
  guides(alpha = "none")  + geom_point(colour = "black",alpha = 0.5) +
  xlab(label1) + ylab(label2) + xlim(-.1, 4.75) + ylim(-.1, 4.75) + 
  theme(legend.position = "none")
# add marginal histograms
ggMarginal(p, type = "histogram",binwidth = 5 / 25)
p2 <-
  ggplot(df2,aes(x = x,y = y)) + geom_point(colour = "white",alpha = 0.5) + stat_ellipse(type = "norm") +
  xlab(label1) + ylab(label2) + xlim(-.1, 4.75) + ylim(-.1, 4.75)
ggMarginal(p2, type = "histogram",binwidth = 5 / 25)



###########################################################
## using Dirichlet distribution
rm(list = ls())

# Choose a condition:
condi <- 2

library(DirichletReg)
library(ggplot2)
library(ggExtra)

dat = read.csv("TestPhase.csv")

## Expression bias
FixDur <- dat$fixDur
Emo <- dat$Emo2
Famil <- dat$Famil
Grp <- dat$Grp
race <- dat$Fctype

# The datasets:
if (condi == 1) {
  x <- FixDur[Emo == "FEAR"]/5
  y <- FixDur[Emo == "HAPPY"]/5
  Grp1 <- Grp[Emo == "FEAR"]
  race1 <- race[Emo == "FEAR"]
  condi2 <- Famil[Emo == "FEAR"]
  label1 <- "FEAR"
  label2 <- "HAPPY"
} else {
  x <- FixDur[Famil == "famil"]/5
  y <- FixDur[Famil == "novel"]/5
  Grp1 <- Grp[Famil == "famil"]
  race1 <- race[Famil == "famil"]
  condi2 <- Emo[Famil == "famil"]
  label1 <- "Famil"
  label2 <- "Novel"
}

z = 1 - x - y
x[x <= 0] = 1 - 0.999999999999999
y[y <= 0] = 1 - 0.999999999999999
z[z <= 0] = 1 - 0.999999999999999

df <- data.frame(outside = z, face1 = x, face2 = y, 
                 Grp = Grp1, race = race1, condi = condi2)
AL <- DR_data(df[,1:3])
plot(AL, cex=.5, a2d=list(colored=TRUE, c.grid=TRUE))

md1 <- DirichReg(AL ~ Grp*race*condi, df,model = "alternative")
summary(md1)
# condi (expression) not significant for main effect and interaction
# fit second model
md2 <- DirichReg(AL ~ Grp*race, df,model = "alternative")
anova(md1,md2)
summary(md2)
confint(md2, exp = TRUE)