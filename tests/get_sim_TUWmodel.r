library(TUWmodel)

for(i in 1:20)
{
    # Open data
    filename <- file.path('input_data', 
                    sprintf('input_data_%0.2d.csv', i))
    data <- read.csv(filename, skip=17, stringsAsFactors=FALSE)
    names(data) <- c('DatesR', 'Precip', 'PotEvap', 'Runoff')

    # date conversion
    data$DatesR = as.POSIXlt(data$DatesR, format='%Y-%m-%d')

    # Parameters for snow melt
    SCF = 1.
    DDF = 1.
    Tr = 1.
    Ts = 0.
    Tm = 0.

    # Generate parameters - soil moisture/response func
    LPrat = runif(1, 0, 1)
    FC = runif(1, 0, 600)
    BETA = runif(1, 0, 20)
    k0 = runif(1, 0, 2)
    k1 = runif(1, 2, 30)
    k2 = runif(1, 30, 250)
    lsuz = runif(1, 1, 100)
    cperc = runif(1, 0, 8)
    bmax = runif(1, 0, 30)
    croute = runif(1, 0, 50)

    # prepare data
    params = c(SCF, DDF, Tr, Ts, Tm, 
                LPrat, FC, BETA,
                k0, k1, k2,
                lsuz, cperc, bmax, 
                croute)

    incon =c(50, 0, 2.5, 2.5)

    # Run model - set air temperature to 30 deg C
    sim = TUWmodel(prec=data$Precip,
        airt=0*data$Precip + 30,
        ep=data$PotEvap,
        area=1,
        param=params,
        incon=incon)
    # Write data
    df = data[c('Precip', 'PotEvap')]
    nms = names(sim)[10:22]
    for(nm in nms){
        df[nm] = as.vector(sim[[nm]])
    }

    filename <- file.path('output_data', 
                    sprintf('HBV_timeseries_%0.2d.csv', i))
    write.csv(df, filename, row.names=FALSE)

    nms = c('LPrat', 'FC', 'BETA', 'K0', 'K1', 'K2', 
                'LSUZ', 'CPERC', 'BMAX', 'CROUTE')
    df = as.data.frame(list(parname=nms, value=params[6:15]))
    filename <- file.path('output_data', 
                    sprintf('HBV_params_%0.2d.csv', i))
    write.csv(df, filename, row.names=FALSE)
}


