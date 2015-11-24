#!/usr/bin/env Rscript

library(DBI)
library(RSQLite)
library(ggplot2)

DB_FILENAME <- 'results_gathered.sqlite'

main <- function()
{
    create_indexes()
    
    variable_names <- load_variable_names()
    
    ccm_data <- load_ccm_data()
    ccm_summ <- summarize_ccm_data(ccm_data, pvalue_threshold = 0.05)
    
    p <- qplot(
        data = ccm_summ,
        x = factor(sigma01), y=factor(sd_proc),
        geom = 'tile', fill=increase_fraction,
        facets = seasonal * identical ~ direction
    )
    
    ggsave(filename='increase_fraction.png', p)
}

# Creates needed DB indexes if not present
create_indexes <- function()
{
    db <- dbConnect(SQLite(), DB_FILENAME)
    
    dbGetQuery(db, 'CREATE INDEX IF NOT EXISTS job_info_index ON job_info (job_id, eps, beta00, sigma01, sd_proc)')
    dbGetQuery(db, 'CREATE INDEX IF NOT EXISTS ccm_increase_index ON ccm_increase (job_id, cause, effect)')
    dbGetQuery(db, 'CREATE INDEX IF NOT EXISTS ccm_correlation_dist_index ON ccm_correlation_dist (job_id, cause, effect, L)')
    
    dbDisconnect(db)
    
    invisible()
}

# Load variable names
load_variable_names <- function()
{
    db <- dbConnect(SQLite(), DB_FILENAME)
    variable_names <- dbGetQuery(db, 'SELECT DISTINCT cause FROM ccm_increase ORDER BY cause')$cause
    dbDisconnect(db)
}

# Constructs a data frame, one row per simulation, for direction cause -> effect with columns:
# eps
# beta00
# sigma01
# sd_proc
# replicate_id
# Lmin
# Lmax
# delays
# pvalue_positive
# pvalue_increase
# mean
# sd
# q0
# q1
# q2_5
# q5
# q25
# q50
# q75
# q95
# q97_5
# q99
# q100
load_ccm_data <- function()
{
    rds_path <- 'ccm_data.Rds'
    
    if(file.exists(rds_path)) {
        return(readRDS(rds_path))
    }
    
    db <- dbConnect(SQLite(), DB_FILENAME)
    
    ccm_data <- dbGetQuery(db, 'SELECT * FROM ccm_increase ORDER BY job_id')
    
    ccm_data$eps <- NA
    ccm_data$beta00 <- NA
    ccm_data$sigma01 <- NA
    ccm_data$sd_proc <- NA
    
    ccm_data$mean <- NA
    ccm_data$sd <- NA
    ccm_data$q0 <- NA
    ccm_data$q1 <- NA
    ccm_data$q2_5 <- NA
    ccm_data$q5 <- NA
    ccm_data$q25 <- NA
    ccm_data$q50 <- NA
    ccm_data$q75 <- NA
    ccm_data$q95 <- NA
    ccm_data$q97_5 <- NA
    ccm_data$q99 <- NA
    ccm_data$q100 <- NA
    
    variable_names <- dbGetQuery(db, 'SELECT DISTINCT cause FROM ccm_increase ORDER BY cause')$cause
    
    for(i in 1:nrow(ccm_data)) {
        row_i <- ccm_data[i,]
        
        job_info <- dbGetPreparedQuery(
            db, 'SELECT * FROM job_info WHERE job_id = ?', data.frame(job_id = row_i$job_id)
        )
        for(colname in c('eps', 'beta00', 'sigma01', 'sd_proc')) {
            ccm_data[i,colname] <- job_info[colname]
        }
        
        ccm_corr_dist <- dbGetPreparedQuery(
            db, 'SELECT * FROM ccm_correlation_dist WHERE job_id = ? AND cause = ? AND effect = ? AND L = ?',
            data.frame(job_id = row_i$job_id, cause = row_i$cause, effect = row_i$effect, L = row_i$Lmax)
        )
        for(colname in c('pvalue_positive', 'mean', 'sd', 'q0', 'q1', 'q2_5', 'q5', 'q25', 'q50', 'q75', 'q95', 'q97_5', 'q99', 'q100')) {
            ccm_data[i, colname] <- ccm_corr_dist[colname]
        }
    }
    
    dbDisconnect(db)
    saveRDS(ccm_data, file = rds_path)
    
    return(ccm_data)
}

# Constructs a summarized data frame with columns:
# eps
# beta00
# sigma01
# sd_proc
# pvalue_positive_mean
# pvalue_positive_sd
# positive_fraction
# pvalue_increase_mean
# pvalue_increase_sd
# increase_fraction
# mean_mean
# median_median
summarize_ccm_data <- function(ccm_data, pvalue_threshold = 0.05)
{
    ccm_summ <- unique(ccm_data[c('eps', 'beta00', 'sigma01', 'sd_proc', 'cause', 'effect')])
    rownames(ccm_summ) <- NULL
    
    ccm_summ$pvalue_positive_mean <- NA
    ccm_summ$pvalue_positive_sd <- NA
    ccm_summ$positive_fraction <- NA
    
    variable_names <- unique(ccm_data$cause)
    level_mat <- outer(variable_names, variable_names, FUN = function(a, b) sprintf('%s causes %s', b, a))
    ccm_summ$direction <- factor(NA, levels = as.character(level_mat))
    print(levels(ccm_summ$direction))
    
    for(i in 1:nrow(ccm_summ)) {
        summ_row <- ccm_summ[i,]
        ccm_data_subset <- ccm_data[
            ccm_data$eps == summ_row$eps &
            ccm_data$beta00 == summ_row$beta00 &
            ccm_data$sigma01 == summ_row$sigma01 &
            ccm_data$sd_proc == summ_row$sd_proc &
            ccm_data$cause == summ_row$cause &
            ccm_data$effect == summ_row$effect,
        ]
        
        ccm_summ[i, 'pvalue_positive_mean'] <- mean(ccm_data_subset$pvalue_positive, na.rm=T)
        ccm_summ[i, 'pvalue_positive_sd'] <- sd(ccm_data_subset$pvalue_positive, na.rm=T)
        ccm_summ[i, 'positive_fraction'] <- mean(ccm_data_subset$pvalue_positive < pvalue_threshold, na.rm=T)
        
        ccm_summ[i, 'pvalue_increase_mean'] <- mean(ccm_data_subset$pvalue_increase, na.rm=T)
        ccm_summ[i, 'pvalue_increase_sd'] <- sd(ccm_data_subset$pvalue_increase, na.rm=T)
        ccm_summ[i, 'increase_fraction'] <- mean(ccm_data_subset$pvalue_increase < pvalue_threshold, na.rm=T)
        
        ccm_summ[i, 'mean_mean'] <- mean(ccm_data_subset$mean, na.rm=T)
        ccm_summ[i, 'median_median'] <- median(ccm_data_subset$q50, na.rm=T)
        
        for(v1 in 1:length(variable_names)) {
            for(v2 in 1:length(variable_names)) {
                if(variable_names[v2] == summ_row$cause && variable_names[v1] == summ_row$effect) {
                    ccm_summ[i, 'direction'] <- level_mat[v1,v2]
                }
            }
        }
    }
    
    ccm_summ$seasonal <- factor(NA, levels = c('nonseasonal', 'seasonal'))
    ccm_summ[ccm_summ$eps == 0.0, 'seasonal'] <- 'nonseasonal'
    ccm_summ[ccm_summ$eps == 0.1, 'seasonal'] <- 'seasonal'
    
    ccm_summ$identical <- factor(NA, levels = c('identical', 'different'))
    ccm_summ[ccm_summ$beta00 == 0.25, 'identical'] <- 'identical'
    ccm_summ[ccm_summ$beta00 == 0.30, 'identical'] <- 'different'
    
    return(ccm_summ)
}

main()
