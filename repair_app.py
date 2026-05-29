"""Repair the corrupted email+syllabus block in app.py."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

with open('app.py', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines before: {len(lines)}")

# Lines 373-422 (1-indexed) = indices 372-421 are corrupted.
# Keep everything before line 373 and after line 422.
good_before = lines[:372]
good_after  = lines[422:]

fix = (
    "\n"
    "            # Email sending\n"
    "            if email_send and email_send.strip():\n"
    "                with st.spinner('Sending study plan email...'):\n"
    "                    syllabus_topics=st.session_state.get('syllabus_topics')\n"
    "                    course_name=st.session_state.get('course_name')\n"
    "                    er=send_study_plan_email(\n"
    "                        recipient_email=email_send.strip(),\n"
    "                        student_name=auth_user['display_name'],\n"
    "                        predicted_score=predicted,\n"
    "                        plan=plan,\n"
    "                        syllabus_topics=syllabus_topics,\n"
    "                        course_name=course_name,\n"
    "                    )\n"
    "                if er['success']:\n"
    "                    st.success('Study plan sent to ' + email_send + '!')\n"
    "                else:\n"
    "                    st.error('Email failed: ' + er['error'])\n"
    "\n"
    "    with tab_syllabus:\n"
    "        st.markdown('### Syllabus Parser')\n"
    "        st.markdown('<p>Upload your syllabus PDF - topics are extracted automatically. Uses Gemini AI when available, falls back to local text parsing otherwise.</p>', unsafe_allow_html=True)\n"
    "        pdf_file=st.file_uploader('Upload Syllabus PDF',type=['pdf'],key='syllabus_pdf')\n"
    "        if pdf_file:\n"
    "            with st.spinner('Analysing your syllabus...'):\n"
    "                result=parse_syllabus_pdf(pdf_file.read())\n"
    "            if result['success']:\n"
    "                st.session_state['syllabus_topics']=result['topics']\n"
    "                st.session_state['course_name']=result['course_name']\n"
    "                src=result.get('source','')\n"
    "                if 'Gemini' in src:\n"
    "                    st.success(result['course_name'] + ' - ' + str(len(result['topics'])) + ' topics via ' + src + '!')\n"
    "                else:\n"
    "                    st.success(result['course_name'] + ' - ' + str(len(result['topics'])) + ' topics extracted (local parser)!')\n"
    "                    st.info('Gemini quota exhausted - used local text parser. Add a fresh GEMINI_API_KEY to .env for best results.')\n"
    "                if result.get('gemini_note'):\n"
    "                    with st.expander('Gemini API note'):\n"
    "                        st.warning(result['gemini_note'])\n"
    "                for unit in result.get('units',[]):\n"
    "                    with st.expander(unit['unit'] + ' (' + str(len(unit['topics'])) + ' topics)'):\n"
    "                        for t in unit['topics']:\n"
    "                            st.markdown('- ' + t)\n"
    "                st.markdown('**All Topics:**')\n"
    "                pills=' '.join(result['topics'])\n"
    "                st.markdown(pills)\n"
    "                st.info('These topics will be included in your next study plan email.')\n"
    "            else:\n"
    "                st.error('Error: ' + result['error'])\n"
    "\n"
)

new_lines = good_before + [fix] + good_after

with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Fixed. New line count: {len(new_lines)}")

# Verify syntax
import ast
try:
    ast.parse(open('app.py', encoding='utf-8').read())
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax error: {e}")
