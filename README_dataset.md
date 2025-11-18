# Vessel Acoustic and AIS Dataset
# overview model 




## Overview dataset


This dataset on MDA contains acoustic and AIS data collected from the Belgian part of the North Sea (BPNS) for vessel activity classification and distance prediction. The data was gathered using hydrophones deployed at two stations, Gardencity and Grafton, positioned near major shipping lanes. 

This dataset has been used for the paper titled:

**Decrop Wout, Deneudt Klaas, Parcerisas Clea, Schall Elena, Debusschere Elisabeth, 2024. Transfer learning for distance classification of marine vessels using underwater sound. Submitted to IEEE.**


The dataset is structured into a table with the following columns:

- file_location: Path to the corresponding 10-second audio file.
- vessel_type: The type of vessel (e.g., Tanker).
- activity: The vessel's reported activity from AIS data (e.g., underway-using-engine).
- SOG: Speed over ground (knots).
- mmsi: Maritime Mobile Service Identity (MMSI) of the vessel.
- longitude: Vessel's longitude at the given timestamp.
- latitude: Vessel's latitude at the given timestamp.
- distance: Distance (km) between the vessel and the hydrophone.
- event_time: Timestamp of the event in UTC.
- station: Hydrophone station where the recording was collected (e.g., Grafton).

## Data Collection

The acoustic data was collected using RESEA 320 RTSys recorders and Colmar GP1190M-LP hydrophones, deployed 1 meter above the seabed on steel mooring frames. The hydrophone stations were positioned in high-traffic maritime areas to maximize vessel detections.

AIS data was sourced from the AIS-Hub data network, providing vessel coordinates, speed, category, and activity. The distance between the hydrophone and vessel was calculated based on these coordinates at each timestamp, ensuring semi-continuous annotations of vessel presence.

## Dataset Creation

The dataset was processed to predict the distance to the nearest vessel using AIS data. A window frame approach was used to determine the closest vessel within each segment:

- 6-minute window frame: Used across most of the dataset.
- 5-minute window frame: Applied in Grafton data from mid-2022 onwards.

A balance was maintained between resolution and accuracy. The smaller window provides more frequent annotations but risks missing vessels due to AIS gaps. The larger window reduces the risk of missing vessels but lowers temporal resolution.

After processing, 27.524 10-second audio segments were extracted over 116 days, including:
- 40 days with overlapping station recordings.
- 76 unique days per station.

Audio files were converted to single-channel, 48 kHz, and segmented into 10-second non-overlapping windows.
Audio Preprocessing and File Naming


Each resulting audio segment was saved using the following filename format:

{output_prefix}/{station_letter}_{deployment_id}_{datetime_str}_{start_delta}_{vessels_information}_{output_postfix}.wav

Filename Components:

- output_prefix: Directory where the output file is stored.
- station_letter: First letter of the station name (always 'g' in this case).
- deployment_id: Identifier of the specific deployment session.
- datetime_str: Timestamp indicating the start time of the segment.
- start_delta: Offset (in seconds) from the beginning of the original audio file, used to trace back the segment's location within the full recording.
- vessels_information: Metadata about vessels present during the segment, including:
  - Vessel type
  - AIS message type description
  - Speed Over Ground (SOG)
- output_postfix: Optional suffix for additional labeling or processing context.

Example filename:
Grafton_15810_train/G_15810_2022-01-20_09-43-17_307-521805_Cargo_underway-using-engine_16-1_3475.wav

This naming structure ensures traceability of each audio segment and includes relevant vessel context for downstream analysis.


## Data Splitting

To ensure independence between training, validation, and testing sets, full-day recordings from one location were only included in a single set. The distribution is as follows:

- Training: 79.4%
- Validation: 10.6%
- Testing: 9.9%

This method ensures no data leakage between sets. The datasplit can be found in the data_split folder

## Important Considerations

- AIS Irregularities: Approximately 8% of data was excluded due to inconsistencies in AIS transmissions.
- Shallow Water Conditions: The BPNS has a maximum depth of 45m, impacting sound propagation.
- Hydrophone Depth: Gardencity (35m), Grafton (23m).

## Folder Structure

The data is organized as follows:
    ├── GardenCity_15811_test
    ├── GardenCity_15811_train
    ├── GardenCity_15811_val
    ├── GardenCity_26981_train
    ├── Grafton_15810_test
    ├── Grafton_15810_train
    ├── Grafton_15810_val
    ├── Grafton_28434_test
    ├── Grafton_28434_train
    ├── Grafton_28434_val
    ├── Grafton_29187_test
    └── Grafton_29187_train

## Coordinates

The following coordinates are used in the dataset for each deployment:

| Deployment Number | Latitude   | Longitude  | Station       |
|-------------------|------------|------------|---------------|
| 15810             | 51.40647   | 2.8185     | bpns-Grafton  |
| 28434             | 51.40666667| 2.819      | bpns-Grafton  |
| 29187             | 51.40666   | 2.819      | bpns-Grafton  |
| 15811             | 51.48645   | 2.304829   | bpns-Gardencity|
| 26981             | 51.48645   | 2.304829   | bpns-Gardencity|


## Citation

If you use this dataset in your work, please cite:

- Parcerisas et al., 2021. LifeWatch Broadband Acoustic Network.  
- Calonge et al., 2024. Seabed Hydrophone Deployments for Vessel Monitoring.  
- AIS-Hub, 2018. AIS Data Network.
- Decrop Wout, Deneudt Klaas, Parcerisas Clea, Schall Elena, Debusschere Elisabeth, 2025. Transfer learning for distance classification of marine vessels using underwater sound. Submitted to IEEE.

