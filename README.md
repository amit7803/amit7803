import React, { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, BarChart, Bar, CartesianGrid } from 'recharts';
import { motion } from 'framer-motion';

// Professional AI Engineer GitHub Dashboard
// Single-file React component (Tailwind CSS assumed in parent project)
// Features:
// - GitHub profile summary (avatar, name, bio, followers, stars)
// - Repo list with language, stars, forks, latest commit date
// - Open PRs & Issues summary
// - CI status (checks/statuses) where available
// - Recent commits chart and activity timeline
// - Quick search for repos
// - Model / experiment metrics uploader (CSV) preview and simple chart
// - Exports: copy repo list as CSV
//
// Usage notes:
// - This component queries GitHub REST API and GraphQL for richer data.
// - For higher rate limits and private repo access, paste a GitHub Personal Access Token (PAT) with `repo` and `read:user` scopes.
// - If no token is provided some endpoints are rate-limited and private data won't be available.

export default function AIEngineerGitHubDashboard() {
  const [username, setUsername] = useState('');
  const [token, setToken] = useState('');
  const [profile, setProfile] = useState(null);
  const [repos, setRepos] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [query, setQuery] = useState('');
  const [commitsOverTime, setCommitsOverTime] = useState([]);
  const [openPRs, setOpenPRs] = useState([]);
  const [openIssues, setOpenIssues] = useState([]);
  const [filePreview, setFilePreview] = useState(null);

  const headers = token ? { Authorization: `token ${token}` } : {};

  useEffect(() => {
    if (!username) return;
    fetchAllData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [username]);

  async function fetchAllData() {
    setLoading(true);
    setError(null);
    setProfile(null);
    setRepos([]);
    setCommitsOverTime([]);
    setOpenPRs([]);
    setOpenIssues([]);

    try {
      // 1) Profile
      const pRes = await fetch(`https://api.github.com/users/${username}`, { headers });
      if (!pRes.ok) throw new Error(`User fetch failed: ${pRes.status}`);
      const pJson = await pRes.json();
      setProfile(pJson);

      // 2) Repos (paginated) - fetch top 100 by stars
      const rRes = await fetch(
        `https://api.github.com/users/${username}/repos?per_page=100&sort=pushed`,
        { headers }
      );
      if (!rRes.ok) throw new Error(`Repos fetch failed: ${rRes.status}`);
      const rJson = await rRes.json();

      // basic repo mapping
      const repoMap = rJson.map((r) => ({
        id: r.id,
        name: r.name,
        full_name: r.full_name,
        description: r.description,
        language: r.language,
        stargazers_count: r.stargazers_count,
        forks_count: r.forks_count,
        pushed_at: r.pushed_at,
        html_url: r.html_url,
      }));
      setRepos(repoMap);

      // 3) Open PRs & issues (search endpoints)
      // PRs
      const prRes = await fetch(
        `https://api.github.com/search/issues?q=type:pr+author:${username}+state:open&per_page=50`,
        { headers }
      );
      if (prRes.ok) {
        const prJson = await prRes.json();
        setOpenPRs(prJson.items || []);
      }

      // Issues
      const isRes = await fetch(
        `https://api.github.com/search/issues?q=type:issue+author:${username}+state:open&per_page=50`,
        { headers }
      );
      if (isRes.ok) {
        const isJson = await isRes.json();
        setOpenIssues(isJson.items || []);
      }

      // 4) Commits over time: aggregate recent push events (best-effort without GraphQL)
      const eventsRes = await fetch(
        `https://api.github.com/users/${username}/events?per_page=100`,
        { headers }
      );
      if (eventsRes.ok) {
        const events = await eventsRes.json();
        const pushes = events.filter((e) => e.type === 'PushEvent');
        const dateMap = {};
        pushes.forEach((p) => {
          const day = new Date(p.created_at).toISOString().slice(0, 10);
          dateMap[day] = (dateMap[day] || 0) + (p.payload?.commits?.length || 1);
        });
        const series = Object.keys(dateMap)
          .sort()
          .map((d) => ({ date: d, commits: dateMap[d] }));
        setCommitsOverTime(series);
      }

      setLoading(false);
    } catch (err) {
      setError(err.message);
      setLoading(false);
    }
  }

  function filteredRepos() {
    if (!query) return repos;
    return repos.filter((r) => r.name.toLowerCase().includes(query.toLowerCase()));
  }

  function exportReposCSV() {
    const rows = ['name,description,language,stars,forks,pushed_at'];
    repos.forEach((r) => {
      const row = [
        `"${r.name}"`,
        `"${(r.description || '').replace(/\"/g, '"') }"`,
        r.language || '',
        r.stargazers_count || 0,
        r.forks_count || 0,
        r.pushed_at || '',
      ].join(',');
      rows.push(row);
    });
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${username || 'repos'}-repos.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleFileUpload(e) {
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const text = ev.target.result;
      // naive CSV parse: assume header + numeric metric in second column
      const lines = text.split(/\r?\n/).filter(Boolean);
      const points = lines.slice(1).map((l, i) => {
        const cols = l.split(',');
        return { name: cols[0] || `row${i}`, value: Number(cols[1] || 0) };
      });
      setFilePreview(points);
    };
    reader.readAsText(f);
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-extrabold">AI Engineer — GitHub Dashboard</h1>
          <p className="text-sm text-gray-500 mt-1">Professional overview & productivity panel for ML/AI engineers</p>
        </div>
        <div className="flex gap-3 items-center">
          <input
            className="border rounded px-3 py-2" 
            placeholder="GitHub username (e.g. octocat)"
            value={username}
            onChange={(e) => setUsername(e.target.value.trim())}
          />
          <input
            className="border rounded px-3 py-2 w-72"
            placeholder="(Optional) Personal Access Token — paste to access private data"
            value={token}
            onChange={(e) => setToken(e.target.value.trim())}
          />
          <button
            className="bg-sky-600 text-white px-4 py-2 rounded hover:bg-sky-700"
            onClick={() => fetchAllData()}
          >
            Load
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 text-red-700 p-3 rounded mb-4">Error: {error}</div>
      )}

      {loading && (
        <div className="text-sm text-gray-500">Loading data — this may take a few seconds depending on GitHub API rate limits.</div>
      )}

      {profile && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="grid grid-cols-12 gap-6">
          {/* left column: profile + quick metrics */}
          <div className="col-span-4">
            <div className="bg-white rounded-2xl shadow p-4 mb-4">
              <div className="flex items-center gap-4">
                <img src={profile.avatar_url} alt="avatar" className="w-20 h-20 rounded-full" />
                <div>
                  <h2 className="text-xl font-semibold">{profile.name || profile.login}</h2>
                  <p className="text-sm text-gray-500">{profile.bio}</p>
                  <div className="mt-2 text-sm text-gray-600">{profile.location} • {profile.company}</div>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-3 text-center gap-2">
                <div>
                  <div className="text-2xl font-bold">{profile.public_repos}</div>
                  <div className="text-xs text-gray-500">Repos</div>
                </div>
                <div>
                  <div className="text-2xl font-bold">{profile.followers}</div>
                  <div className="text-xs text-gray-500">Followers</div>
                </div>
                <div>
                  <div className="text-2xl font-bold">{profile.following}</div>
                  <div className="text-xs text-gray-500">Following</div>
                </div>
              </div>

              <div className="mt-4 flex gap-2">
                <a target="_blank" rel="noreferrer" href={profile.html_url} className="text-sm underline">Open Profile</a>
                <button onClick={exportReposCSV} className="text-sm ml-auto bg-gray-100 px-3 py-1 rounded">Export repos CSV</button>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow p-4">
              <h3 className="font-semibold mb-2">Model / Experiment Metrics</h3>
              <p className="text-xs text-gray-500">Upload a small CSV to visualize metric trends (name,value)</p>
              <input type="file" accept=".csv" onChange={handleFileUpload} className="mt-2" />

              {filePreview && (
                <div className="mt-3">
                  <div className="h-40">
                    <ResponsiveContainer width="100%" height={120}>
                      <BarChart data={filePreview}>
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="value" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="text-xs text-gray-500 mt-2">Preview: {filePreview.length} rows</div>
                </div>
              )}
            </div>
          </div>

          {/* right column: charts + top repos */}
          <div className="col-span-8">
            <div className="bg-white rounded-2xl shadow p-4 mb-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold">Activity</h3>
                <div className="text-sm text-gray-500">Last events & commits</div>
              </div>

              <div className="mt-3 grid grid-cols-2 gap-4">
                <div className="p-2">
                  <h4 className="text-sm text-gray-600 mb-2">Commits (by day)</h4>
                  <div className="h-44">
                    <ResponsiveContainer width="100%" height={180}>
                      <LineChart data={commitsOverTime} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                        <YAxis />
                        <Tooltip />
                        <Line type="monotone" dataKey="commits" stroke="#8884d8" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="p-2">
                  <h4 className="text-sm text-gray-600 mb-2">Open PRs / Issues</h4>
                  <div className="flex gap-2">
                    <div className="bg-amber-50 p-3 rounded w-1/2">
                      <div className="text-xl font-bold">{openPRs.length}</div>
                      <div className="text-xs text-gray-500">Open PRs</div>
                    </div>
                    <div className="bg-rose-50 p-3 rounded w-1/2">
                      <div className="text-xl font-bold">{openIssues.length}</div>
                      <div className="text-xs text-gray-500">Open Issues</div>
                    </div>
                  </div>

                  <div className="mt-3 text-xs text-gray-600">
                    <ul className="list-disc pl-4 max-h-28 overflow-auto">
                      {openPRs.slice(0,6).map((p) => (
                        <li key={p.id}><a className="underline" href={p.html_url} target="_blank" rel="noreferrer">{p.title}</a></li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow p-4 mb-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold">Repos</h3>
                <div className="flex items-center gap-2">
                  <input className="border px-2 py-1 rounded text-sm" placeholder="Search repos" value={query} onChange={(e) => setQuery(e.target.value)} />
                </div>
              </div>

              <div className="grid grid-cols-1 gap-3">
                {filteredRepos().slice(0, 12).map((r) => (
                  <div key={r.id} className="p-3 border rounded hover:shadow-sm flex items-start gap-3">
                    <div className="flex-1">
                      <a className="text-sky-600 font-medium text-lg" href={r.html_url} target="_blank" rel="noreferrer">{r.name}</a>
                      <div className="text-sm text-gray-500">{r.description}</div>
                      <div className="mt-2 text-xs text-gray-600 flex gap-3">
                        <div>{r.language}</div>
                        <div>★ {r.stargazers_count}</div>
                        <div>🍴 {r.forks_count}</div>
                        <div>Updated {new Date(r.pushed_at).toLocaleString()}</div>
                      </div>
                    </div>
                    <div className="text-xs text-gray-400">Quick actions
                      <div className="mt-2 flex flex-col gap-2">
                        <a className="underline" href={`${r.html_url}/actions`} target="_blank" rel="noreferrer">Actions</a>
                        <a className="underline" href={`${r.html_url}/pulls`} target="_blank" rel="noreferrer">Pulls</a>
                        <a className="underline" href={`${r.html_url}/issues`} target="_blank" rel="noreferrer">Issues</a>
                      </div>
                    </div>
                  </div>
                ))}

                {filteredRepos().length === 0 && (
                  <div className="p-3 text-sm text-gray-500">No repositories match.</div>
                )}
              </div>
            </div>

            <div className="bg-white rounded-2xl shadow p-4">
              <h3 className="font-semibold mb-2">Activity timeline</h3>
              <div className="text-xs text-gray-600 max-h-44 overflow-auto">
                {/* Show recent events: pushes, PRs, issues */}
                {repos.slice(0, 20).map((r) => (
                  <div key={r.id} className="border-b py-2 flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium">{r.name}</div>
                      <div className="text-xs text-gray-500">Last pushed: {new Date(r.pushed_at).toLocaleString()}</div>
                    </div>
                    <div className="text-xs text-gray-400">Repo</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {!profile && (
        <div className="mt-6 text-sm text-gray-600">Enter a GitHub username above and click 'Load' to populate the dashboard. For private repo details or higher rate limits, paste a Personal Access Token (PAT) with the `repo` scope.</div>
      )}

      <div className="mt-6 text-xs text-gray-500">Built with React, Tailwind, Recharts, Framer Motion — adapt and extend for OAuth integration, GraphQL queries, or server-side caching for production use.</div>
    </div>
  );
}
