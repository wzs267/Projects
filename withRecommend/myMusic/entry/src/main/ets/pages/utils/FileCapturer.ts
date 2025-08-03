'use strict';
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */

import { ICapturerInterface } from './ICapturerInterface'
import {fileIo} from '@kit.CoreFileKit';
import { Util } from './Util'
import { AsrConstants } from './AsrConstants';


const TAG = 'FileCapturer';

/**
 * File collector tool
 */
export default class FileCapturer implements ICapturerInterface {
  /**
   * Whether the audio is being written
   */
  private mIsWriting: boolean = false;

  /**
   * File Path
   */
  private mFilePath: string = '';

  /**
   * Open File Object
   */
  private mFile: fileIo.File = null;

  /**
   * Indicates whether the file can be read.
   */
  private mIsReadFile: boolean = true;

  /**
   * Audio Data Callback Method
   */
  private mDataCallBack: (data: ArrayBuffer) => void = null;

  /**
   * Setting the File Path
   * @param filePath
   */
  public setFilePath(filePath: string) {
    this.mFilePath = filePath;
  }

  async init(dataCallBack: (data: ArrayBuffer) => void) {
    if (null != this.mDataCallBack) {
      return;
    }
    this.mDataCallBack = dataCallBack;
    if (!fileIo.accessSync(this.mFilePath)) {
      return
    }
    console.error(TAG, "init start ");
  }

  async start() {
    console.error(TAG, "a");
    try {
      if (this.mIsWriting || null == this.mDataCallBack) {
        return;
      }
      console.error(TAG, "b");
      this.mIsWriting = true;
      this.mIsReadFile = true;
      this.mFile = fileIo.openSync(this.mFilePath, fileIo.OpenMode.READ_ONLY);
      console.error(TAG, "c");
      let buf: ArrayBuffer = new ArrayBuffer(AsrConstants.SEND_SIZE);
      let offset: number = 0;
      let count = 0;
      console.error(TAG, "d");
      while (AsrConstants.SEND_SIZE == fileIo.readSync(this.mFile.fd, buf, {
        offset: offset
      }) && this.mIsReadFile) {
        console.error(TAG, "e");
        this.mDataCallBack(buf);
        ++count;
        await Util.countDownLatch(1);
        offset = offset + AsrConstants.SEND_SIZE;
      }
    } catch (e) {
      console.error(TAG, "read file error " + e);
    } finally {
      if (null != this.mFile) {
        fileIo.closeSync(this.mFile);
      }
      this.mIsWriting = false;
    }
  }

  stop() {
    if (null == this.mDataCallBack) {
      return;
    }
    try {
      this.mIsReadFile = false;
    } catch (e) {
      console.error(TAG, "read file error " + e);
    }
  }

  release() {
    if (null == this.mDataCallBack) {
      return;
    }
    try {
      this.mDataCallBack = null;
      this.mIsReadFile = false;
    } catch (e) {
      console.error(TAG, "read file error " + e);
    }
  }
}
