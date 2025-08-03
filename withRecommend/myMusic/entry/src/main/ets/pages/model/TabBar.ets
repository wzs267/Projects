export interface  TabBarModel{
  title: string;
  name: string;
  index: number;
  icon: ResourceStr;
}
// TabBar 的数据实现
export const tabsData :TabBarModel[] = [
  {
    title: '首页',
    name: 'home',
    index:0,
    icon: $r('app.media.ic_recommend')
  },
  {
    title: '我的歌单',
    name: 'likeSongList',
    index:1,
    icon: $r('app.media.ic_logo')
  },
  {
    title: '评论',
    name: 'comment',
    index:2,
    icon: $r('app.media.ic_comment')
  },
  {
    title: '我的',
    name: 'mine',
    index:3,
    icon: $r('app.media.ic_mine')
  }
];