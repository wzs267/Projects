'use strict';

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */


/**
 * Code value corresponding to the listener callback event.
 */
export enum LISTENER_CODE {
  /**
   * Initialize
   */
  METHOD_ON_INIT = 1,

  /**
   * Callback when human voice is detected
   */
  METHOD_ON_BEGINNING_OF_SPEECH = 2,

  /**
   * Return the volume energy value in real time.
   */
  METHOD_ON_RMS_CHANGED = 3,

  /**
   * The user ends speaking.
   */
  METHOD_ON_END_OF_SPEECH = 4,

  /**
   * Network or identification error
   */
  METHOD_ON_ERROR = 5,

  /**
   * Triggered when the recognition scenario is dictation
   */
  METHOD_ON_PARTIAL_RESULTS = 6,

  /**
   * Identification result
   */
  METHOD_ON_RESULTS = 7,

  /**
   * Trigger semantic vad or multimode vad
   */
  METHOD_ON_SUB_TEXT = 8,

  /**
   * The current session identification ends.
   */
  METHOD_ON_END = 9,

  /**
   * The word map update is complete.
   */
  METHOD_ON_LEXICON_UPDATED = 10,

  /**
   * End of updating parameters.
   */
  METHOD_ON_UPDATE_PARAMS = 11
}

/**
 * Constant class
 */
export class AsrConstants {
  /**
   * VAD front-end point duration
   */
  public static readonly ASR_VAD_FRONT_WAIT_MS: string = "vad_front_wait_ms";

  /**
   * Endpoint Duration After VAD
   */
  public static readonly ASR_VAD_END_WAIT_MS: string = "vad_end_wait_ms";

  /**
   * ASR Timeout Interval
   */
  public static readonly ASR_TIMEOUT_THRESHOLD_MS: string = "timeout_threshold_ms";

  /**
   * Size of audio sent each time
   */
  public static readonly SEND_SIZE: number = 1280;
}



