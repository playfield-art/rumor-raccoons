# Rumor API Status code documentation

## Status codes
| CODE | MEANING                                                                 |
|------|-------------------------------------------------------------------------|
| 200  | OK                                                                      |
|      |                                                                         |
| 4xx  | Client Error - the request contains bad syntax or cannot be fulfilled   |
|      |                                                                         |
| 5xx  | Server Error - the server failed to fulfill an apparently valid request |
| 520  | Default Server Error - see detail for info                              |
|      |                                                                         |
| 530  | OpenAI Error - authentication error                                     |
| 531  | OpenAI Error - model load error                                         |
| 532  | OpenAI Error - invalid request error                                    |
| 533  | OpenAI Error - connect error                                            |
|      |                                                                         |
| 580  | Database Error - connection error                                       |
| 581  | Database Error - error while reading collection                         |
| 582  | Database Error - no data found in collection                            |
| 583  | Database Error - write error                                            |
| 584  | Database Error - error while parsing data in database                   |


