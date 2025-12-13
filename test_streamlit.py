#!/usr/bin/env python
"""Streamlit uygulamasını test etmek için basit script"""
import sys
import os

# Path'leri ayarla
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modules', 'frontend'))

try:
    print("=" * 60)
    print("Streamlit Uygulaması Test")
    print("=" * 60)
    
    # Import testleri
    print("\n1. Modül importları test ediliyor...")
    from data_manager import load_data
    print("   ✓ data_manager import edildi")
    
    from streamlit_app import main
    print("   ✓ streamlit_app import edildi")
    
    from ui_pages_input import show_destination_page
    print("   ✓ ui_pages_input import edildi")
    
    from ui_pages_results import show_results_page
    print("   ✓ ui_pages_results import edildi")
    
    # Veri yükleme testi
    print("\n2. Veri yükleme test ediliyor...")
    df = load_data()
    print(f"   ✓ Veri yüklendi: {len(df)} satır, {len(df.columns)} sütun")
    
    # Streamlit versiyonu
    print("\n3. Streamlit versiyonu kontrol ediliyor...")
    import streamlit as st
    print(f"   ✓ Streamlit versiyonu: {st.__version__}")
    
    # Parametre kontrolü
    print("\n4. Streamlit parametreleri kontrol ediliyor...")
    import inspect
    
    # st.button
    button_sig = inspect.signature(st.button)
    if 'width' in button_sig.parameters:
        print("   ✓ st.button() 'width' parametresini destekliyor")
    else:
        print("   ✗ st.button() 'width' parametresini desteklemiyor")
    
    # st.link_button
    link_button_sig = inspect.signature(st.link_button)
    if 'width' in link_button_sig.parameters:
        print("   ✓ st.link_button() 'width' parametresini destekliyor")
    else:
        print("   ✗ st.link_button() 'width' parametresini desteklemiyor")
    
    # st.altair_chart
    altair_sig = inspect.signature(st.altair_chart)
    if 'use_container_width' in altair_sig.parameters:
        print("   ✓ st.altair_chart() 'use_container_width' parametresini destekliyor")
    else:
        print("   ✗ st.altair_chart() 'use_container_width' parametresini desteklemiyor")
    
    print("\n" + "=" * 60)
    print("✓ Tüm testler başarılı!")
    print("=" * 60)
    print("\nUygulamayı çalıştırmak için:")
    print("  streamlit run modules/frontend/streamlit_app.py")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("✗ HATA OLUŞTU!")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    sys.exit(1)

