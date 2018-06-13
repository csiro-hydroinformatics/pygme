library(airGR)

# Get X3 value from command line
# low value of X3 (<5) trigger the oscillating behaviour
args <- commandArgs()
X3 <- as.numeric(args[6])

# Open data
filename <- file.path('data', '606002_inputs.csv')
data <- read.csv(filename, stringsAsFactors=FALSE)
names(data) <- c('DatesR', 'Precip', 'PotEvap')

filename <- file.path('data', '606002_params.csv')
params <- read.csv(filename, stringsAsFactors=FALSE)

# date conversion
data$DatesR = as.POSIXlt(data$DatesR, format='%Y-%m-%d')

# Model inputs
inputs <- CreateInputsModel(FUN_MOD=RunModel_GR4J, 
            DatesR = data$DatesR,
            Precip = data$Precip,
            PotEvap = data$PotEvap)

nval <- length(data$DatesR)
indwarmup <- seq(1, 366*3)
indrun <- seq(366*3+1, nval)
boolcrit <- data$Runoff[indrun] >= 0

# ------ GR4J ------------------
#gr4j_params <- c(params[,'X1'], params[,'X2'], 
#                    params[,'X3'], params[,'X4'])
gr4j_params <- c(params[,'X1'], params[,'X2'], 
                    X3, params[,'X4'])

IniStates <- c(gr4j_params[1]/2, gr4j_params[3]*0.3)

runoptions <- CreateRunOptions(FUN_MOD = RunModel_GR4J,
                           InputsModel = inputs,
                           IndPeriod_Run = indrun,
                           IniStates = NULL, 
                           IniResLevels = NULL, 
                           IndPeriod_WarmUp = indwarmup)

# Run model
outputs <- RunModel_GR4J(InputsModel = inputs,
                            RunOptions = runoptions, 
                            Param = gr4j_params)

# Write data
nm = names(outputs)[2:19]
df = as.data.frame(outputs[nm])
filename <- file.path('output_data', 'GR4J_timeseries_606002.csv')
write.csv(df, filename, row.names=FALSE)

