function default_measurements(mc::DQMC, model::ZCTModel)
    Dict(
        :conf => ConfigurationMeasurement(mc, model),
        :Greens => GreensMeasurement(mc, model),
        :BosonEnergy => BosonEnergyMeasurement(mc, model),
        :Magnetization => MagnetizationMeasurement(mc, model),
        :Superconductivity => SuperconductivityMeasurement(mc, model)
        # :Chirality => ChiralityMeasurement(mc, model)
    )
end
