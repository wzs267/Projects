'use strict';
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */

/**
 * Collector port
 *
 * @since 2023-09-08
 */
export interface ICapturerInterface {
  /**
   * Initialize
   * @param dataCallBack Audio Data Callback Method
   */
  init(dataCallBack: (data: ArrayBuffer) => void);

  /**
   * start
   */
  start();

  /**
   * stop
   */
  stop();

  /**
   * release
   */
  release();
}