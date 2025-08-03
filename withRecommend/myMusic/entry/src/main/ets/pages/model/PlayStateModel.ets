import { SongModel } from './AppModel';
// 播放器通知页面的数据结构
export interface PlayStateType {
  pic: string
  name: string
  url: string
  playIndex: number
  time: number
  duration: number
  isPlay: boolean
  playMode: 'auto' | 'repeat' | 'random'
  playList: SongModel[]
}
export class PlayStateModel implements  PlayStateType{

  pic: string;
  name: string;
  url: string;
  playIndex: number;
  time: number;
  duration: number;
  isPlay: boolean;
  playMode: 'auto' | 'repeat' | 'random';
  playList: SongModel[];

  constructor(playState:PlayStateType ) {
    this.pic = playState.pic
    this.name = playState.name
    this.time = playState.time
    this.url = playState.url
    this.playIndex = playState.playIndex
    this.duration = playState.duration
    this.isPlay = playState.isPlay
    this.playMode = playState.playMode
    this.playList = playState.playList
  }
}

export const defaultState:PlayStateType = new PlayStateModel({
  pic:'',
  name:'',
  url:'',
  playIndex:0,
  time:0,
  duration:0,
  isPlay:false,
  playMode:'auto',
  playList:[]
})
