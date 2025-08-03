'use strict';

/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */

/**
 * Tools
 *
 * @since 2023-08-25
 */
export class Util {
  /**
   * Waiter
   * @param count
   */
  public static async countDownLatch(count: number) {
    while (count > 0) {
      await this.sleep(40);
      count--;
    }
  }

  /**
   * Sleeping
   * @param ms
   * @returns
   */
  private static sleep(ms: number) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}