import os
import pandas as pd
import streamlit as st
from models.selection import select_top_beneficiaries
from models.distribution import calculate_aid_distribution
from models.classification import classify_data

st.set_page_config(page_title="Charity System", layout="wide")

with st.container(border=True):
    st.info("""
    ## **Welcome to our Smart Charity Management System**  
    ### *This system helps charities classify beneficiaries, select those most deserving, and distribute donations fairly.*
    """)

st.subheader("Choose an Action")
card1, card2, card3 = st.columns(3)

with card1:
    with st.container(border=True):
        st.info("## 1. Classification")
        st.write("#### *Classify beneficiaries into categories: A, B or C* ")
        st.space(70)
        run_class = st.button("Run Classification", use_container_width=True)

with card2:
    with st.container(border=True):
        st.info("## 2. Select Beneficiaries")
        st.write("#### *Select top N beneficiaries based on need*")
        num_people = st.number_input("", min_value=1, value=10)
        select_ben = st.button("Select Beneficiaries", use_container_width=True)

with card3:
    with st.container(border=True):
        st.info("## 3. Aid Distribution")
        st.markdown("#### *Distribute budget fairly among top beneficiaries* ")
        budget = st.number_input("", min_value=100, value=10000)
        calc_dist = st.button("Calculate Distribution", use_container_width=True)

st.divider()

uploaded_file = st.file_uploader("Upload Data", type="csv")

tab1, tab2, tab3, tab4 = st.tabs(["Data", "Classification", "Selected Beneficiaries", "Aid Distribution"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    with tab1:
        st.subheader("Original Uploaded Data")
        st.dataframe(df, use_container_width=True)

    with tab2:
        st.subheader("Classification Results")
        if run_class:
            temp_path = os.path.join("uploads", uploaded_file.name)
            try:
                #ينشئ مجلدًا اسمه uploads إذا لم يكن موجودًا.
                os.makedirs("uploads", exist_ok=True)
                #يفتح ملفًا جديدًا للكتابة.
                #"w" = Write (كتابة)
                #"b" = Binary (ملف ثنائي مثل CSV أو Excel أو صورة)
                with open(temp_path, "wb") as f:

                    f.write(uploaded_file.getbuffer())

                with st.spinner("Classifying data... "):
                    # 🎯 استلام الـ DataFrame بالكامل جاهز وفيه عمود التصنيف
                    cat_df = classify_data(temp_path)

                st.success("Classification Completed")
                st.dataframe(cat_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Click 'Run Classification' above to see results")

    with tab3:
        st.subheader("Selected Top Beneficiaries")
        if select_ben:
            temp_path = os.path.join("uploads", uploaded_file.name)
            try:
                os.makedirs("uploads", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Selecting Top Beneficiaries... "):
                    top_df = select_top_beneficiaries(temp_path, int(num_people))

                st.success(f"Selected top {len(top_df)} beneficiaries based on Need Score!")
                st.dataframe(top_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Enter 'Number of People' and click 'Select Beneficiaries' above")

    with tab4:
        st.subheader("Aid Distribution Breakdown")
        if calc_dist:
            temp_path = os.path.join("uploads", uploaded_file.name)
            try:
                os.makedirs("uploads", exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Calculating fair distribution... "):
                    # 1. جلب الأكثر احتياجاً من الموديل أولاً
                    top_df = select_top_beneficiaries(temp_path, int(num_people))
                    # 2. حساب وتوزيع الميزانية عليهم
                    distributed_df = calculate_aid_distribution(top_df, float(budget))

                st.success(f"Successfully distributed EGP {float(budget):,.2f} among top {len(distributed_df)} beneficiaries")
                
                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Distributed Budget", f"{distributed_df['Allocated_Aid (EGP)'].sum():,.2f} EGP")
                m2.metric("Max Single Aid", f"{distributed_df['Allocated_Aid (EGP)'].max():,.2f} EGP")
                m3.metric("Min Single Aid", f"{distributed_df['Allocated_Aid (EGP)'].min():,.2f} EGP")

                # عرض الجدول
                cols_to_show = ["Name", "Gender", "Age", "Need_Score", "Need_Ratio", "Allocated_Aid (EGP)"]
                available_cols = [c for c in cols_to_show if c in distributed_df.columns]
                st.dataframe(distributed_df[available_cols] if available_cols else distributed_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Enter 'Total Budget' and click 'Calculate Distribution' above.")

else:
    for tab in [tab1, tab2, tab3, tab4]:
        with tab:
            st.info("No result — Please upload a CSV file first")