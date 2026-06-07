// JARVIS System 4 - Dashboard Logic

const API_BASE = 'http://localhost:8080/api';

// Format time
function updateClock() {
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    document.getElementById('time-display').textContent = timeString;
}

setInterval(updateClock, 1000);
updateClock();

// Fetch Data
async function fetchDashboardData() {
    try {
        const [academicRes, roadmapRes, statsRes, focusRes] = await Promise.all([
            fetch(`${API_BASE}/academic`).catch(() => null),
            fetch(`${API_BASE}/roadmap`).catch(() => null),
            fetch(`${API_BASE}/stats`).catch(() => null),
            fetch(`${API_BASE}/focus`).catch(() => null)
        ]);

        if (academicRes && academicRes.ok) updateAcademic(await academicRes.json());
        if (roadmapRes && roadmapRes.ok) updateRoadmap(await roadmapRes.json());
        if (statsRes && statsRes.ok) updateStats(await statsRes.json());
        if (focusRes && focusRes.ok) updateFocus(await focusRes.json());

    } catch (e) {
        console.error("API Fetch Error:", e);
    }
}

function updateAcademic(data) {
    if (!data) return;

    // CGPA
    if (data.cgpa && data.cgpa.current) {
        const current = data.cgpa.current;
        const target = data.cgpa.target || 8.5;
        document.getElementById('current-cgpa').textContent = current;
        document.getElementById('target-cgpa-display').textContent = target;
        
        // Calculate stroke dasharray for circle (max 100)
        const percent = (current / 10.0) * 100;
        document.getElementById('cgpa-circle').setAttribute('stroke-dasharray', `${percent}, 100`);
    }

    // Exam Alerts
    const examModePill = document.getElementById('exam-mode-pill');
    if (data.exam_mode) {
        examModePill.style.display = 'flex';
    } else {
        examModePill.style.display = 'none';
    }

    // Upcoming exams list
    const examsList = document.getElementById('upcoming-exams-list');
    if (data.alerts && data.alerts.length > 0) {
        examsList.innerHTML = '';
        data.alerts.forEach(exam => {
            const el = document.createElement('div');
            el.className = 'exam-item';
            el.innerHTML = `
                <span class="exam-subject">${exam.subject}</span>
                <span class="exam-days">${exam.days_left} days</span>
            `;
            examsList.appendChild(el);
        });
    }
}

function updateRoadmap(data) {
    if (!data) return;

    const phase = data.current_phase || 'foundation';
    document.getElementById('ms-current-phase').textContent = phase.charAt(0).toUpperCase() + phase.slice(1) + " Phase";
    
    if (data.phase_progress && data.phase_progress[phase]) {
        const percent = data.phase_progress[phase].percent || 0;
        document.getElementById('ms-phase-percent').textContent = `${percent}%`;
        document.getElementById('ms-progress-bar').style.width = `${percent}%`;

        // Update milestones
        const list = document.getElementById('pending-milestones');
        const pending = data.phase_progress[phase].pending_milestones || [];
        
        if (pending.length > 0) {
            list.innerHTML = '';
            pending.slice(0, 3).forEach(m => {
                const el = document.createElement('div');
                el.className = 'milestone-item';
                el.innerHTML = `
                    <i class="fa-regular fa-circle"></i>
                    <span>${m.description}</span>
                `;
                list.appendChild(el);
            });
        } else {
            list.innerHTML = `<p class="empty-state">All milestones completed!</p>`;
        }
    }
}

function updateStats(data) {
    if (!data) return;

    if (data.daily) {
        document.getElementById('stat-study-hours').textContent = data.daily.study_hours.toFixed(1);
        document.getElementById('stat-pomodoros').textContent = data.daily.pomodoros;
        document.getElementById('stat-tasks').textContent = data.daily.tasks_completed;
    }

    if (data.weekly) {
        document.getElementById('weekly-study').textContent = `${data.weekly.study_hours_total.toFixed(1)} / ${data.weekly.study_target_hours} h`;
        document.getElementById('weekly-papers').textContent = data.weekly.papers_read;
        document.getElementById('weekly-tasks').textContent = data.weekly.tasks_completed;
    }
}

function updateFocus(data) {
    if (!data) return;
    
    const focusPill = document.getElementById('focus-pill');
    if (data.status !== "no_session" && data.phase === "work") {
        focusPill.style.display = 'flex';
        // Add time remaining to pill
        focusPill.innerHTML = `<i class="fa-solid fa-stopwatch"></i> ${Math.floor(data.phase_remaining_minutes)} MIN FOCUS`;
    } else {
        focusPill.style.display = 'none';
    }
}

// Initial fetch and poll every 10 seconds
fetchDashboardData();
setInterval(fetchDashboardData, 10000);
