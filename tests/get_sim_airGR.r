library(airGR)

for(i in 1:20)
{
    # Open data
    filename <- file.path('input_data', 
                    sprintf('input_data_%0.2d.csv', i))
    data <- read.csv(filename, skip=17, stringsAsFactors=FALSE)
    names(data) <- c('DatesR', 'Precip', 'PotEvap', 'Runoff')

    filename <- file.path('input_data', 
                    sprintf('input_data_monthly_%0.2d.csv', i))
    datam <- read.csv(filename, skip=17, stringsAsFactors=FALSE)
    names(datam) <- c('DatesR', 'Precip', 'PotEvap', 'Runoff')

    # date conversion
    data$DatesR = as.POSIXlt(data$DatesR, format='%Y-%m-%d')
    datam$DatesR = as.POSIXlt(datam$DatesR, format='%Y-%m-%d')

    # Model inputs daily
    inputs <- CreateInputsModel(FUN_MOD=RunModel_GR6J, 
                DatesR = data$DatesR,
                Precip = data$Precip,
                PotEvap = data$PotEvap)

    nval <- length(data$DatesR)
    indwarmup <- seq(1, 366*3)
    indrun <- seq(366*3+1, nval)
    boolcrit <- data$Runoff[indrun] >= 0

    # Model inputs monthly
    inputsm <- CreateInputsModel(FUN_MOD=RunModel_GR2M, 
                DatesR = datam$DatesR,
                Precip = datam$Precip,
                PotEvap = datam$PotEvap)

    nvalm <- length(datam$DatesR)
    indwarmupm <- seq(1, 12*3)
    indrunm <- seq(12*3+1, nvalm)
    boolcritm <- datam$Runoff[indrunm] >= 0


    # ------ GR4J ------------------
    runoptions <- CreateRunOptions(FUN_MOD = RunModel_GR4J,
                               InputsModel = inputs,
                               IndPeriod_Run = indrun,
                               IniStates = NULL, 
                               IniResLevels = NULL, 
                               IndPeriod_WarmUp = indwarmup)

    # Calibration
    crit <- CreateInputsCrit(FUN_CRIT = ErrorCrit_KGE, 
                                InputsModel = inputs,
                                RunOptions = runoptions,
                                BoolCrit = boolcrit,
                                Qobs = data$Runoff[indrun])
    
    calib <- CreateCalibOptions(FUN_MOD = RunModel_GR4J,
                            FUN_CALIB = Calibration_Michel)

    outputscalib <- Calibration_Michel(InputsModel = inputs, 
                            RunOptions = runoptions,
                            InputsCrit = crit, 
                            CalibOptions = calib,
                            FUN_MOD = RunModel_GR4J, 
                            FUN_CRIT = ErrorCrit_KGE)

    gr4j_params <- outputscalib$ParamFinalR
    gr4j_params[1] = max(1, gr4j_params[1])
    gr4j_params[3] = max(1, gr4j_params[3])

    # Run model
    outputs <- RunModel_GR4J(InputsModel = inputs,
                                RunOptions = runoptions, 
                                Param = gr4j_params)

    # Write data
    nm = names(outputs)[2:19]
    df = as.data.frame(outputs[nm])
    df$Qobs <- data$Runoff[indrun]
    filename <- file.path('output_data', 
                    sprintf('GR4J_timeseries_%0.2d.csv', i))
    write.csv(df, filename, row.names=FALSE)

    df = as.data.frame(gr4j_params)
    filename <- file.path('output_data', 
                    sprintf('GR4J_params_%0.2d.csv', i))
    write.csv(df, filename, row.names=FALSE)

    # ------ GR6J ------------------
    runoptions <- CreateRunOptions(FUN_MOD = RunModel_GR6J,
                               InputsModel = inputs,
                               IndPeriod_Run = indrun,
                               IniStates = NULL, 
                               IniResLevels = NULL, 
                               IndPeriod_WarmUp = indwarmup)

    # Calibration
    crit <- CreateInputsCrit(FUN_CRIT = ErrorCrit_KGE, 
                                InputsModel = inputs,
                                RunOptions = runoptions,
                                BoolCrit = boolcrit,
                                Qobs = data$Runoff[indrun])
    
    calib <- CreateCalibOptions(FUN_MOD = RunModel_GR6J,
                            FUN_CALIB = Calibration_Michel)

    outputscalib <- Calibration_Michel(InputsModel = inputs, 
                            RunOptions = runoptions,
                            InputsCrit = crit, 
                            CalibOptions = calib,
                            FUN_MOD = RunModel_GR6J, 
                            FUN_CRIT = ErrorCrit_KGE)

    gr6j_params <- outputscalib$ParamFinalR
    gr6j_params[1] = max(1, gr6j_params[1])
    gr6j_params[3] = max(1, gr6j_params[3])
    gr6j_params[6] = max(0.1, gr6j_params[6])

    # Run model
    outputs <- RunModel_GR6J(InputsModel = inputs,
                                RunOptions = runoptions, 
                                Param = gr6j_params)

    # Write data
    nm = names(outputs)[2:21]
    df = as.data.frame(outputs[nm])
    df$Qobs <- data$Runoff[indrun]
    filename <- file.path('output_data', 
                    sprintf('GR6J_timeseries_%0.2d.csv', i))
    write.csv(df, filename, row.names=FALSE)

    df = as.data.frame(gr6j_params)
    filename <- file.path('output_data', 
                    sprintf('GR6J_params_%0.2d.csv', i))
    write.csv(df, filename, row.names=FALSE)

    # ------ GR2M ------------------
    runoptionsm <- CreateRunOptions(FUN_MOD = RunModel_GR2M,
                               InputsModel = inputsm,
                               IndPeriod_Run = indrunm,
                               IniStates = NULL, 
                               IniResLevels = NULL, 
                               IndPeriod_WarmUp = indwarmupm)

    # Calibration
    critm <- CreateInputsCrit(FUN_CRIT = ErrorCrit_KGE, 
                                InputsModel = inputsm,
                                RunOptions = runoptionsm,
                                BoolCrit = boolcritm,
                                Qobs = datam$Runoff[indrunm])
    
    calibm <- CreateCalibOptions(FUN_MOD = RunModel_GR2M,
                            FUN_CALIB = Calibration_Michel)

    outputscalibm <- Calibration_Michel(InputsModel = inputsm, 
                            RunOptions = runoptionsm,
                            InputsCrit = critm, 
                            CalibOptions = calibm,
                            FUN_MOD = RunModel_GR2M, 
                            FUN_CRIT = ErrorCrit_KGE)

    gr2m_params <- outputscalibm$ParamFinalR

    # Run model
    outputsm <- RunModel_GR2M(InputsModel = inputsm,
                                RunOptions = runoptionsm, 
                                Param = gr2m_params)

    # Write data
    nm = names(outputsm)[2:10]
    df = as.data.frame(outputsm[nm])
    df$Qobs <- datam$Runoff[indrunm]
    filename <- file.path('output_data', 
                    sprintf('GR2M_timeseries_%0.2d.csv', i))
    write.csv(df, filename, row.names=FALSE)

    df = as.data.frame(gr2m_params)
    filename <- file.path('output_data', 
                    sprintf('GR2M_params_%0.2d.csv', i))
    write.csv(df, filename, row.names=FALSE)

}


