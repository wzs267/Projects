#!/usr/bin/env python3
"""
检查音符生成范围问题
"""

def analyze_note_generation():
    """分析音符生成范围"""
    # 模拟当前参数
    duration = 140.0  # 秒
    tempo = 104.2     # BPM
    
    # 当前的计算方式
    beats_per_second = tempo / 60
    total_measures_old = int((duration * beats_per_second) / 4)
    
    print(f"🎵 歌曲参数:")
    print(f"   时长: {duration}秒")
    print(f"   BPM: {tempo}")
    print(f"   每秒拍数: {beats_per_second:.2f}")
    
    print(f"\n❌ 当前错误计算:")
    print(f"   小节数: {total_measures_old}")
    print(f"   覆盖时间: {(total_measures_old * 4) / beats_per_second:.1f}秒")
    
    # 正确的计算方式
    total_beats = duration * beats_per_second
    total_measures_correct = int(total_beats / 4)
    
    print(f"\n✅ 正确计算:")
    print(f"   总拍数: {total_beats:.1f}")
    print(f"   小节数: {total_measures_correct}")
    print(f"   覆盖时间: {(total_measures_correct * 4) / beats_per_second:.1f}秒")
    
    # 计算每小节生成的音符数量
    note_density = 0.22
    notes_per_measure = 4 * 4 * note_density  # 每小节4拍，每拍4个细分，乘以密度
    total_estimated_notes = total_measures_correct * notes_per_measure
    
    print(f"\n🎼 音符生成估算:")
    print(f"   Note密度: {note_density}")
    print(f"   每小节音符数: {notes_per_measure:.1f}")
    print(f"   预计总音符数: {total_estimated_notes:.0f}")
    
    return total_measures_correct

if __name__ == "__main__":
    correct_measures = analyze_note_generation()
    print(f"\n💡 我们需要生成 {correct_measures} 个小节的音符，而不是当前的少量小节！")
